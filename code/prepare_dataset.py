#!/usr/bin/env python3
"""Convert PGN games into a training dataset matching the Maia3 input format.

Output (memmap-friendly .npy files + metadata json):
  tokens.npy     (N, 64, 97) float16 - board-history tokens identical to what
                 maia3.uci feeds the model at inference: 12*history one-hot board
                 channels tokenized with each board's own side-to-move, padded at
                 the front with the earliest position, plus an always-zero ponder
                 column (every released checkpoint uses include_time_info=False).
  move_idx.npy   (N,) uint16  - model-space (mirrored-board) move vocabulary index
                 of the move played. Computed from the mirrored board directly, so
                 black promotions map to correct a7a8q-style entries.
  legal_mask.npy (N, 544) uint8 - packbits of the 4352-length legal-move mask,
                 also derived from the mirrored board.
  value_idx.npy  (N,) uint8   - 0=loss 1=draw 2=win for the side to move (WDL head).
  elo.npy        (N, 2) int32 - (self_elo, oppo_elo) of the side to move.
  game_id.npy    (N,) uint32 - game index per position (for split by game).
  train_idx.npy / val_idx.npy - position indices per split.
  meta.json      - dataset config, counts, skip stats, elo summary.

Usage:
  .venv/bin/python code/prepare_dataset.py data/chesscom/rapid_dataset_ALL.pgn \
      --accounts TsukinoLuna Sasayaku --out data/datasets/rapid_style
"""

from __future__ import annotations

import argparse
import chess
import chess.pgn
import json
import math
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maia3.dataset import get_historical_tokens, tokenize_board  # noqa: E402
from maia3.utils import get_all_possible_moves, seed_everything  # noqa: E402

ALL_MOVES = get_all_possible_moves()
ALL_MOVES_DICT = {m: i for i, m in enumerate(ALL_MOVES)}
N_MOVES = len(ALL_MOVES)

HISTORY = 8            # matches every released Maia3 checkpoint
INCLUDE_TIME_INFO = False
TOKEN_DIM = 12 * HISTORY + 1  # 97; last column is the (unused) ponder label


class TokenCfg:
    def __init__(self, history: int, include_time_info: bool):
        self.history = history
        self.include_time_info = include_time_info


TOKEN_CFG = TokenCfg(HISTORY, INCLUDE_TIME_INFO)


def int_elo(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 1500


def square_mirror(sq: int) -> int:
    return sq ^ 56


def mirrored_move(board, move) -> chess.Move:
    """Same move, expressed on the white-perspective mirrored board."""
    if board.turn == chess.WHITE:
        return chess.Move(move.from_square, move.to_square, promotion=move.promotion)
    return chess.Move(square_mirror(move.from_square), square_mirror(move.to_square),
                      promotion=move.promotion)


def account_of(game, accounts):
    white = game.headers.get("White", "").lower()
    black = game.headers.get("Black", "").lower()
    if accounts:
        if white in accounts:
            return white, True
        if black in accounts:
            return black, True
        return None, False
    return "all", True


def build_game_samples(game, accounts):
    """Return (samples, account, skip_reason). samples is a list of sample dicts."""
    account, involved = account_of(game, accounts)
    if not involved:
        return None, None, "not-your-game"
    result = game.headers.get("Result", "*")
    if result not in {"1-0", "0-1", "1/2-1/2"}:
        return None, None, "unsupported-result"
    variant = game.headers.get("Variant", "Standard")
    if variant != "Standard":
        return None, None, "variant"

    white_elo, black_elo = int_elo(game.headers.get("WhiteElo")), int_elo(game.headers.get("BlackElo"))
    if [game.headers.get("WhiteElo"), game.headers.get("BlackElo")].count(None) or \
       "?" in str(game.headers.get("WhiteElo", "0")) + str(game.headers.get("BlackElo", "0")):
        defaulted_elo = True
    else:
        defaulted_elo = False

    white_wins = result == "1-0"
    draw = result == "1/2-1/2"

    samples = []
    board = game.board()
    node = game
    tokenized = []  # tokenized board for every ply prefix of this game

    while node.variations:
        move = node.variations[0].move
        if move is None:
            break

        m_board = board.copy(stack=False) if board.turn == chess.WHITE else board.mirror()
        m_move = mirrored_move(board, move)
        label = ALL_MOVES_DICT.get(m_move.uci())
        if label is None:
            return None, None, f"vocab-miss({m_move.uci()})"

        tokenized.append(tokenize_board(board))
        window = deque(tokenized[-HISTORY:])
        tokens = get_historical_tokens(window, TOKEN_CFG,
                                       base=0.0, inc=0.0,
                                       clk_left_before=0.0, clk_ponder=0.0)
        assert tokens.shape == (64, TOKEN_DIM), tokens.shape

        legal = torch.zeros(N_MOVES, dtype=torch.bool)
        for legal_move in m_board.legal_moves:
            idx = ALL_MOVES_DICT.get(legal_move.uci())
            if idx is not None:
                legal[idx] = True
        if not bool(legal[label]):
            return None, None, f"illegal-label({m_move.uci()})"

        if board.turn == chess.WHITE:
            self_elo, oppo_elo = white_elo, black_elo
            won = white_wins
        else:
            self_elo, oppo_elo = black_elo, white_elo
            won = result == "0-1"
        value_idx = 1 if draw else (2 if won else 0)

        samples.append({
            "tokens": tokens.to(torch.float16),
            "move_idx": label,
            "legal": legal,
            "value_idx": value_idx,
            "self_elo": self_elo,
            "oppo_elo": oppo_elo,
        })
        board.push(move)
        node = node.variations[0]

    if not samples:
        return None, None, "empty"
    return samples, account, ("elo-defaulted" if defaulted_elo else None)


def iter_games(paths):
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            while True:
                try:
                    game = chess.pgn.read_game(f)
                except Exception as e:  # noqa: BLE001
                    print(f"warning: skipping corrupt game: {e}", file=sys.stderr, flush=True)
                    continue
                if game is None:
                    break
                yield game


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pgn", nargs="+", help="PGN file(s)")
    ap.add_argument("--accounts", nargs="*", default=None,
                    help="Only keep games involving these player(s) (case-insensitive)")
    ap.add_argument("--out", default="data/datasets/style")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--val-min", type=int, default=30)
    ap.add_argument("--max-games", type=int, default=None, help="Optional cap (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="maia3-79m",
                    help="Alias recorded in meta.json only; input format is the same "
                         "for every released Maia3 checkpoint")
    args = ap.parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    accounts = {a.lower() for a in args.accounts} if args.accounts else None

    # -------- pass 1: count usable positions per game (cheap, no tokenization) ------
    t0 = time.time()
    skips = Counter()
    game_infos = []  # (gamebook index is game order in files) placeholders: n_pos unknown yet
    # Instead of replicating the skip logic, just run the full builder in pass 1 too,
    # keeping only count metadata; tokenization cost is small (~1-2 min for 100k pos).
    counts = []
    for game in iter_games(args.pgn):
        if args.max_games and len(counts) >= args.max_games:
            break
        samples, _account, reason = build_game_samples(game, accounts)
        if samples is None:
            skips[reason.split("(")[0]] += 1
            continue
        counts.append(len(samples))
    total_positions = int(sum(counts))
    n_games = len(counts)
    if n_games == 0:
        sys.exit(f"No usable games found in {args.pgn}")
    print(f"pass 1: {n_games} games, {total_positions} usable positions, "
          f"skipped={dict(skips)} ({time.time()-t0:.1f}s)")

    # -------- allocate outputs --------
    tokens_mm = np.lib.format.open_memmap(out_dir / "tokens.npy", mode="w+",
                                          dtype=np.float16,
                                          shape=(total_positions, 64, TOKEN_DIM))
    legal_mm = np.lib.format.open_memmap(out_dir / "legal_mask.npy", mode="w+",
                                         dtype=np.uint8,
                                         shape=(total_positions, math.ceil(N_MOVES / 8)))
    move_idx_mm = np.lib.format.open_memmap(out_dir / "move_idx.npy", mode="w+",
                                            dtype=np.uint16, shape=(total_positions,))
    value_idx_mm = np.lib.format.open_memmap(out_dir / "value_idx.npy", mode="w+",
                                             dtype=np.uint8, shape=(total_positions,))
    elo_mm = np.lib.format.open_memmap(out_dir / "elo.npy", mode="w+",
                                       dtype=np.int32, shape=(total_positions, 2))
    game_id_mm = np.lib.format.open_memmap(out_dir / "game_id.npy", mode="w+",
                                           dtype=np.uint32, shape=(total_positions,))

    # -------- pass 2: write --------
    t0 = time.time()
    pos = 0
    written_counts = []  # positions actually written per game (should equal pass 1)
    elo_accum = []
    account_of_gid = []
    gid = 0
    for game in iter_games(args.pgn):
        if args.max_games and gid >= args.max_games:
            break
        samples, _account, reason = build_game_samples(game, accounts)
        if samples is None:
            skips[f"pass2.{reason.split('(')[0]}"] += 1
            continue
        start, end = pos, pos + len(samples)
        for s in samples:
            tokens_mm[pos] = s["tokens"].numpy()          # (64, 97) fp16
            legal_mm[pos] = np.packbits(s["legal"].numpy())  # 4352 bits -> 544 B
            move_idx_mm[pos] = s["move_idx"]
            value_idx_mm[pos] = s["value_idx"]
            elo_mm[pos] = (s["self_elo"], s["oppo_elo"])
            game_id_mm[pos] = gid
            elo_accum.append((s["self_elo"], s["oppo_elo"]))
            pos += 1
        written_counts.append(end - start)
        account_of_gid.append(_account)
        gid += 1

    if pos != total_positions:
        print(f"warning: pass1 counted {total_positions} but wrote {pos} positions",
              file=sys.stderr)
    print(f"pass 2: wrote {pos} positions ({time.time()-t0:.1f}s)")

    for mm in (tokens_mm, legal_mm, move_idx_mm, value_idx_mm, elo_mm, game_id_mm):
        mm.flush()

    # -------- splits: random games -> train/val --------
    rs = np.random.RandomState(args.seed)
    n_val = max(args.val_min, int(round(n_games * args.val_fraction)))
    n_val = min(n_val, gid - 1)
    val_games = set(rs.choice(gid, size=n_val, replace=False).tolist())
    gids = np.zeros(pos, dtype=np.uint32)
    # rebuild gid array quickly by slicing the memmap we just wrote
    gids[:] = game_id_mm[:]
    train_pos = np.flatnonzero(~np.isin(gids, list(val_games))).astype(np.uint32)
    val_pos = np.flatnonzero(np.isin(gids, list(val_games))).astype(np.uint32)
    np.save(out_dir / "train_idx.npy", train_pos)
    np.save(out_dir / "val_idx.npy", val_pos)

    elos = np.array(elo_accum, dtype=np.float32)
    meta = {
        "pgn_files": [str(p) for p in args.pgn],
        "accounts": args.accounts,
        "games": gid,
        "positions": pos,
        "train_positions": int(train_pos.size),
        "val_positions": int(val_pos.size),
        "val_games_n": n_val,
        "skips": dict(skips),
        "input_format": {
            "history": HISTORY,
            "include_time_info": INCLUDE_TIME_INFO,
            "token_dim": TOKEN_DIM,
            "move_vocab": N_MOVES,
            "note": "tokens/masks/labels are computed in model space (board mirrored "
                    "when black to move); black promotions encoded from the mirrored "
                    "board so labels align with get_all_possible_moves()",
        },
        "elo_summary": {
            "self_mean": float(elos[:, 0].mean()) if pos else None,
            "oppo_mean": float(elos[:, 1].mean()) if pos else None,
            "self_min": int(elos[:, 0].min()) if pos else None,
            "self_max": int(elos[:, 0].max()) if pos else None,
        },
        "model_alias_recorded": args.model,
        "accounts_by_game": [{"gid": i, "account": a} for i, a in
                             list(enumerate(account_of_gid))[:20]],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

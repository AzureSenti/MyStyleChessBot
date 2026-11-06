#!/usr/bin/env python3
"""Dataset builder v4 — implements quytactrain.md §2-4.

§3.1 game-level filters:
  standard only · rapid kept (w=1.0), blitz kept (w=0.3), bullet dropped ·
  player's own rating >= --min-rating (else dropped, = §3.3 option A) ·
  |rating gap| <= --max-gap · game >= --min-ply plies ·
  "abandoned"/"aborted" termination dropped · dedup by Link header.

§3.2 move-level filters (applied to the ACCOUNT's moves; dropped everywhere):
  forced moves (exactly 1 legal reply) · spent < --premove-sec (reflex/premove) ·
  time-starved: clk < max(--low-time-sec, --low-time-frac * base_time) ·
  decided positions (|SF eval| >= --decided-cp) kept at weight x0.5.

Sampling weight out: weight.npy = tc_weight * (0.5 if decided else 1).
Split (§4): games sorted by UTC date → 70% train / 10% val / 15% test —
never split by ply; near-duplicate openings across splits reported.

Outputs: tokens/legal_mask/move_idx/value_idx/elo/game_id (+weight.npy, ply.npy,
train_idx/val_idx/test_idx) + meta.json with every filter's counters.

Usage:
  .venv/bin/python code/prepare_dataset_v2.py data/chesscom/all_rapid_blitz.pgn \
      --accounts TsukinoLuna Sasayaku --out data/datasets/style_v4
"""
from __future__ import annotations

import argparse
import chess
import chess.engine
import chess.pgn
import json
import math
import re
import subprocess
import sys
import threading
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
HISTORY = 8
INCLUDE_TIME_INFO = False
TOKEN_DIM = 12 * HISTORY + 1
DEFAULT_SF = str(Path(__file__).resolve().parent.parent / ".tools" / "stockfish" / "stockfish")
CLK_RE = re.compile(r"clk\s+(\d+):(\d+):(\d+)(?:\.(\d+))?")

_tl = threading.local()
_ENGINES, _ENGINES_LOCK = [], threading.Lock()


class TokenCfg:
    def __init__(self, history, include_time_info):
        self.history = history
        self.include_time_info = include_time_info


TOKEN_CFG = TokenCfg(HISTORY, INCLUDE_TIME_INFO)


def int_elo(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def square_mirror(sq):
    return sq ^ 56


def mirrored_move(board, move):
    if board.turn == chess.WHITE:
        return chess.Move(move.from_square, move.to_square, promotion=move.promotion)
    return chess.Move(square_mirror(move.from_square), square_mirror(move.to_square),
                      promotion=move.promotion)


def parse_timecontrol(tc):
    if not tc or tc.startswith("1/"):
        return None, None
    parts = tc.split("+")
    try:
        return float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0
    except (ValueError, IndexError):
        return None, None


def tc_class(base):
    if base is None:
        return "rapid"
    if base >= 600:
        return "rapid"
    if base >= 180:
        return "blitz"
    return "bullet"


def clk_seconds(comment):
    m = CLK_RE.search(comment or "")
    if not m:
        return None
    h, mn, s, frac = m.groups()
    sec = int(h) * 3600 + int(mn) * 60 + int(s)
    if frac:
        sec += float("0." + frac)
    return sec


def spawn_engine(sf_path):
    eng = chess.engine.SimpleEngine.popen_uci([sf_path],
                                              stderr=subprocess.DEVNULL, setpgrp=True)
    eng.configure({"Threads": 1, "Hash": 64})
    with _ENGINES_LOCK:
        _ENGINES.append(eng)
    return eng


def close_all_engines():
    with _ENGINES_LOCK:
        for e in _ENGINES:
            try:
                e.quit()
            except Exception:
                pass
        _ENGINES.clear()


def sf_eval(board, eng, depth):
    """Parent-position eval from mover's perspective."""
    try:
        s = eng.analyse(board, chess.engine.Limit(depth=depth),
                        info=chess.engine.INFO_SCORE)["score"].relative
        if s.is_mate():
            return 10000 if s.mate() > 0 else -10000
        return s.score()
    except chess.engine.EngineError:
        return None


def tl_engine(sf_path):
    """thread-local SF engine (one per worker thread)."""
    eng = getattr(_tl, "engine", None)
    if eng is None:
        eng = spawn_engine(sf_path)
        _tl.engine = eng
    return eng


def game_sig(g, n_plies):
    return " ".join(mv.uci() for mv in list(g.mainline_moves())[:n_plies])


def game_id_of(g):
    """Chess.com games carry the numeric id in the Link header (giống filter_games.py)."""
    for key in ("Link", "ChesscomID", "Site"):
        v = g.headers.get(key, "") or ""
        if v and "/" in v:
            return v.rstrip("/").split("/")[-1]
    return v or "unknown"


def load_quality_csv(path):
    """CSV của filter_games.py → {game_id: kept_bool}. Entry thiếu → mất an toàn (drop)."""
    import csv as _csv
    keep = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            keep[str(row["game_id"])] = (row.get("kept", "0") == "1")
    return keep


# ---------------------------------------------------------------- §3.1 game filters
def game_info(g, args, accounts, seen_links):
    h = g.headers
    if h.get("Variant", "Standard").strip() != "Standard":
        return None, "variant"
    if h.get("Result", "*") not in {"1-0", "0-1", "1/2-1/2"}:
        return None, "result"
    term = (h.get("Termination", "") or "").lower()
    if "abandoned" in term or "aborted" in term:
        return None, "abandoned"
    if sum(1 for _ in g.mainline_moves()) < args.min_ply:
        return None, "short"
    link = h.get("Link", "") or ""
    if link:
        if link in seen_links:
            return None, "dup-link"
        seen_links.add(link)
    w_elo, b_elo = int_elo(h.get("WhiteElo")), int_elo(h.get("BlackElo"))
    if w_elo is None or b_elo is None:
        return None, "elo-missing"
    acc = {a.lower() for a in (accounts or [])}
    w, b = h.get("White", "").lower(), h.get("Black", "").lower()
    is_white = w in acc
    if accounts and not is_white and b not in acc:
        return None, "not-your-game"
    self_elo, oppo_elo = (w_elo, b_elo) if is_white else (b_elo, w_elo)
    base, _inc = parse_timecontrol(h.get("TimeControl", ""))
    klass = tc_class(base)
    if klass == "bullet":
        return None, "bullet"
    # v5: min-rating CHỈ áp dụng cho rapid (blitz giữ nguyên theo yêu cầu user)
    if klass == "rapid" and self_elo < args.min_rating:
        return None, f"rating<{args.min_rating}({klass})"
    if abs(self_elo - oppo_elo) > args.max_gap:
        return None, f"rating-gap>{args.max_gap}"
    # v5: gate chất lượng theo CSV (chess.com accuracy; loss-only gate được nướng sẵn
    # vào cột kept của filter_games.py) — thiếu entry → drop an toàn, đếm riêng
    if getattr(args, "_quality", None) is not None:
        gid = game_id_of(g)
        if gid not in args._quality:
            return None, "quality-csv-missing"
        if not args._quality[gid]:
            return None, "quality-accuracy-drop"
    return {
        "tc": klass, "base": base,
        "white_elo": w_elo, "black_elo": b_elo,
        "self_elo": self_elo, "oppo_elo": oppo_elo, "is_white": is_white,
        "start_utc": h.get("UTCDate", "2000.01.01") + " " + h.get("UTCTime", "00:00:00"),
    }, None


# ---------------------------------------------------------------- §3.2 move filters
def analyze_game(g, info, args, sf_path, use_sf):
    h = g.headers
    result = h["Result"]
    white_wins = result == "1-0"
    draw = result == "1/2-1/2"

    board = g.board()
    node = g
    samples = []
    skips = Counter()
    prev_clk = {}
    tokenized = []
    base = info["base"] or 600.0
    eng = tl_engine(sf_path) if use_sf else None

    while node.variations:
        child = node.variations[0]
        move = child.move
        if move is None:
            break
        mover = board.turn
        mover_is_acc = (mover == chess.WHITE) == info["is_white"]
        if not mover_is_acc:
            board.push(move)
            node = child
            continue

        clk = clk_seconds(child.comment)
        last = prev_clk.get(mover, base)
        spent = (last - clk) if (clk is not None and last > 0) else None
        prev_clk[mover] = clk if clk is not None else last
        m_ply = board.ply() + 1          # ply of the move being played (1-based)

        # v4.1: bỏ nước khai khỏi dữ liệu (bot chơi khai bằng sách riêng)
        if m_ply <= args.drop_open_ply:
            skips["opening"] += 1
            board.push(move); node = child
            continue

        # ngưỡng cạn giờ: tàn cuộc (≥ --end-ply) chặt hơn (bỏ nước "chạy đồng hồ")
        low_thresh = max(args.low_time_sec, args.low_time_frac * base)
        if m_ply >= args.end_ply:
            low_thresh = max(low_thresh, args.eg_low_time_sec)

        # legal idx list (mirrored space)
        m_board = board.copy(stack=False) if mover == chess.WHITE else board.mirror()
        legal_idx = []
        for mvl in m_board.legal_moves:
            i = ALL_MOVES_DICT.get(mvl.uci())
            if i is not None:
                legal_idx.append(i)

        if len(legal_idx) == 1:
            skips["forced"] += 1
            board.push(move); node = child
            continue
        if spent is not None and spent < args.premove_sec:
            skips["premove"] += 1
            board.push(move); node = child
            continue
        if clk is not None and clk < low_thresh:
            skips["low-time"] += 1
            board.push(move); node = child
            continue

        m_move = mirrored_move(board, move)
        label = ALL_MOVES_DICT.get(m_move.uci())
        if label is None:
            return None, skips, f"vocab-miss({m_move.uci()})"

        tokenized.append(tokenize_board(board))
        window = deque(tokenized[-HISTORY:])
        tokens = get_historical_tokens(window, TOKEN_CFG, base=0.0, inc=0.0,
                                       clk_left_before=0.0, clk_ponder=0.0)

        legal = torch.zeros(N_MOVES, dtype=torch.bool)
        for i in legal_idx:
            legal[i] = True
        if not bool(legal[label]):
            return None, skips, f"illegal-label({m_move.uci()})"

        self_elo, oppo_elo = ((info["white_elo"], info["black_elo"])
                              if mover == chess.WHITE else
                              (info["black_elo"], info["white_elo"]))
        if draw:
            value_idx = 1
        elif (mover == chess.WHITE) == white_wins:
            value_idx = 2
        else:
            value_idx = 0

        w = 1.0 if info["tc"] == "rapid" else args.weight_blitz
        if eng is not None:
            ev = sf_eval(board, eng, args.sf_depth)
            if ev is not None and abs(ev) >= args.decided_cp:
                w *= 0.5
                skips["decided"] += 1

        samples.append({
            "tokens": tokens.to(torch.float16),
            "move_idx": label,
            "legal": legal,
            "value_idx": value_idx,
            "self_elo": self_elo,
            "oppo_elo": oppo_elo,
            "w": np.float32(w),
            "ply": board.ply() + 1,
        })
        board.push(move)
        node = child

    if not samples:
        return None, skips, "no-kept-moves"
    return samples, skips, None


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
    ap.add_argument("pgn", nargs="+")
    ap.add_argument("--accounts", nargs="+", required=True)
    ap.add_argument("--out", default="data/datasets/style_v4")
    ap.add_argument("--model", default="maia3-79m")
    ap.add_argument("--seed", type=int, default=42)
    # §3.1 game level
    ap.add_argument("--min-rating", type=int, default=1250,
                    help="v5: chỉ áp dụng cho rapid (blitz không lọc rating)")
    ap.add_argument("--quality-csv", default=None,
                    help="CSV của filter_games.py (cột game_id, kept). Ván không entry "
                         "hoặc kept=0 → bị bỏ (loss-only accuracy gate nướng sẵn trong CSV)")
    ap.add_argument("--max-gap", type=int, default=400)
    ap.add_argument("--min-ply", type=int, default=10)
    # §3.2 move level
    ap.add_argument("--premove-sec", type=float, default=1.0)
    ap.add_argument("--low-time-sec", type=float, default=30.0)
    ap.add_argument("--low-time-frac", type=float, default=0.10)
    ap.add_argument("--drop-open-ply", type=int, default=0,
                    help="v4.1: bỏ nước khai có ply <= giá trị (mặc định 0 = giữ)")
    ap.add_argument("--end-ply", type=int, default=60,
                    help="ply bắt đầu giai đoạn tàn cuộc cho ngưỡng thời gian riêng")
    ap.add_argument("--eg-low-time-sec", type=float, default=60.0,
                    help="tàn cuộc: bỏ nước khi clk < giây này (áp lực thời gian)")
    ap.add_argument("--decided-cp", type=int, default=800)
    ap.add_argument("--sf-depth", type=int, default=10)
    ap.add_argument("--no-sf", action="store_true", help="skip decided-position tagging")
    ap.add_argument("--sf-workers", type=int, default=8)
    # §3.1 blitz weight & §4 split
    ap.add_argument("--weight-blitz", type=float, default=0.3)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--clash-ply", type=int, default=12, help="opening near-dup window")
    ap.add_argument("--max-games", type=int, default=None, help="smoke-test cap")
    args = ap.parse_args()
    seed_everything(args.seed)

    if args.sf_depth and not Path(DEFAULT_SF).exists():
        print("warning: Stockfish not found — decided-position tagging disabled", file=sys.stderr)
        args.no_sf = True
    use_sf = not args.no_sf

    args._quality = None
    if args.quality_csv:
        args._quality = load_quality_csv(args.quality_csv)
        print(f"quality-csv: {len(args._quality)} entries, "
              f"kept={sum(args._quality.values())}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    seen_links = set()
    games_skipped = Counter()
    moves_filtered = Counter()
    games_meta = []       # (gid, info dict)
    game_samples = []     # list of sample lists, in file order
    sigs = []             # opening signature per kept game

    games = list(iter_games(args.pgn))
    if args.max_games:
        games = games[:args.max_games]
    print(f"loaded {len(games)} games ({time.time()-t0:.1f}s)", flush=True)

    ordered_results = [None] * len(games)
    from concurrent.futures import ThreadPoolExecutor

    def work(i):
        g = games[i]
        info, reason = game_info(g, args, args.accounts, seen_links)
        if info is None:
            return i, None, reason, None, None
        samples, skips, mreason = analyze_game(g, info, args, DEFAULT_SF, use_sf)
        sig = game_sig(g, args.clash_ply) if samples is not None else None
        return i, samples, (mreason if samples is None else None), skips, (info, sig)

    with ThreadPoolExecutor(max_workers=args.sf_workers) as pool:
        for i, samples, reason, skips, meta in pool.map(work, range(len(games))):
            if samples is None:
                games_skipped[reason.split("(")[0]] += 1
            else:
                info, sig = meta
                gid = len(game_samples)
                game_samples.append(samples)
                games_meta.append((gid, info))
                sigs.append(sig)
                for k, v in skips.items():
                    moves_filtered[k] += v

    close_all_engines()
    n_games = len(game_samples)
    total_positions = sum(len(s) for s in game_samples)
    if n_games == 0:
        sys.exit(f"No usable games found in {args.pgn}")
    print(f"pass 1: {n_games} games kept, {total_positions} positions, "
          f"skipped={dict(games_skipped)}, moves_filtered={dict(moves_filtered)} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # -------- allocate & write --------
    tokens_mm = np.lib.format.open_memmap(out_dir / "tokens.npy", mode="w+",
                                          dtype=np.float16, shape=(total_positions, 64, TOKEN_DIM))
    legal_mm = np.lib.format.open_memmap(out_dir / "legal_mask.npy", mode="w+",
                                         dtype=np.uint8, shape=(total_positions, math.ceil(N_MOVES / 8)))
    move_idx_mm = np.lib.format.open_memmap(out_dir / "move_idx.npy", mode="w+", dtype=np.uint16,
                                            shape=(total_positions,))
    value_idx_mm = np.lib.format.open_memmap(out_dir / "value_idx.npy", mode="w+", dtype=np.uint8,
                                             shape=(total_positions,))
    elo_mm = np.lib.format.open_memmap(out_dir / "elo.npy", mode="w+", dtype=np.int32,
                                       shape=(total_positions, 2))
    game_id_mm = np.lib.format.open_memmap(out_dir / "game_id.npy", mode="w+", dtype=np.uint32,
                                           shape=(total_positions,))
    weight_mm = np.lib.format.open_memmap(out_dir / "weight.npy", mode="w+", dtype=np.float32,
                                          shape=(total_positions,))
    ply_mm = np.lib.format.open_memmap(out_dir / "ply.npy", mode="w+", dtype=np.uint16,
                                       shape=(total_positions,))

    pos = 0
    for gid, samples in enumerate(game_samples):
        for s in samples:
            tokens_mm[pos] = s["tokens"].numpy()
            legal_mm[pos] = np.packbits(s["legal"].numpy())
            move_idx_mm[pos] = s["move_idx"]
            value_idx_mm[pos] = s["value_idx"]
            elo_mm[pos] = (s["self_elo"], s["oppo_elo"])
            game_id_mm[pos] = gid
            weight_mm[pos] = s["w"]
            ply_mm[pos] = s["ply"]
            pos += 1
    assert pos == total_positions
    for mm in (tokens_mm, legal_mm, move_idx_mm, value_idx_mm, elo_mm,
               game_id_mm, weight_mm, ply_mm):
        mm.flush()

    # -------- §4 chronological split 70/10/15 --------
    order = sorted(range(n_games), key=lambda gidx: games_meta[gidx][1]["start_utc"])
    n_train = int(round(n_games * args.train_frac))
    n_val = int(round(n_games * args.val_frac))
    train_g = set(order[:n_train])
    val_g = set(order[n_train:n_train + n_val])
    test_g = set(order[n_train + n_val:])
    gids = game_id_mm[:].copy()
    split_idx = {}
    for name, gset in (("train", train_g), ("val", val_g), ("test", test_g)):
        split_idx[name] = np.flatnonzero(np.isin(gids, list(gset))).astype(np.uint32)
        np.save(out_dir / f"{name}_idx.npy", split_idx[name])

    # near-duplicate opening report (§4)
    sig_counts = Counter(s for s in sigs if s)
    dup_across = sum(c - 1 for s, c in sig_counts.items() if c > 1)
    crossing = 0
    sig_groups = {}
    for gid, sig in enumerate(sigs):
        if sig:
            sig_groups.setdefault(sig, []).append(gid)
    for gid_list in sig_groups.values():
        gsets = {"train" if gi in train_g else "val" if gi in val_g else "test" for gi in gid_list}
        if len(gsets) > 1:
            crossing += 1

    splits_dates = {}
    for name, gset in (("train", train_g), ("val", val_g), ("test", test_g)):
        dates = sorted(games_meta[gi][1]["start_utc"] for gi in gset)
        splits_dates[name] = {
            "games": len(gset),
            "from": dates[0] if dates else None,
            "to": dates[-1] if dates else None,
            "positions": int(split_idx[name].size),
        }

    w_arr = weight_mm[:]
    meta = {
        "pgn_files": [str(p) for p in args.pgn],
        "accounts": args.accounts,
        "games": n_games,
        "positions": total_positions,
        "rules": {
            "min_rating": args.min_rating, "min_rating_applies": "rapid-only",
            "quality_csv": args.quality_csv,
            "max_gap": args.max_gap, "min_ply": args.min_ply,
            "premove_sec": args.premove_sec, "low_time_sec": args.low_time_sec,
            "low_time_frac": args.low_time_frac, "decided_cp": args.decided_cp,
            "sf_depth": args.sf_depth if use_sf else None,
            "weight_blitz": args.weight_blitz, "drop_premove": True, "drop_forced": True,
            "drop_lowtime": True, "dedup_link": True, "drop_bullet": True,
            "shuffle": False,
        },
        "splits": {name: splits_dates[name] for name in ("train", "val", "test")},
        "games_skipped": dict(games_skipped),
        "moves_filtered": dict(moves_filtered),
        "weight_summary": {
            "mean": float(w_arr.mean()),
            "n_blitz_w0.3": int((w_arr == args.weight_blitz).sum()),
            "n_blitz_decided_w0.15": int((w_arr == args.weight_blitz * 0.5).sum()),
            "n_rapid_w1.0": int((w_arr == 1.0).sum()),
            "n_rapid_decided_w0.5": int((w_arr == 0.5).sum()),
            "n_decided_moves": int(moves_filtered.get("decided", 0)),
        },
        "opening_near_dup": {
            "dup_pairs_at_first_%d_plys" % args.clash_ply: int(dup_across),
            "groups_with_cross_split_same_sig": int(crossing),
            "note": "count of identical opening signatures shared between splits — "
                    "these are memorized lines, report separately in eval (§4, §9)",
        },
        "input_format": {
            "history": HISTORY, "include_time_info": INCLUDE_TIME_INFO,
            "token_dim": TOKEN_DIM, "move_vocab": N_MOVES,
            "note": "tokens/masks/labels in model space (mirrored when black to move)",
        },
        "elo_summary": {
            "self_mean": float(elo_mm[:, 0][split_idx["train"]].mean()),
            "oppo_mean": float(elo_mm[:, 1][split_idx["train"]].mean()),
            "self_min": int(elo_mm[:, 0][split_idx["train"]].min()),
            "self_max": int(elo_mm[:, 0][split_idx["train"]].max()),
        },
        "model_alias_recorded": args.model,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()

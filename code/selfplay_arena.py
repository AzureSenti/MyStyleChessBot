#!/usr/bin/env python3
"""
Arena: generate games of fine-tuned Maia3 (AzuPre) vs limited-strength Stockfish.

Purpose: create a large volume of games where the BOT plays the user's style,
sampled against varied opponents, to fine-tune further / stabilize the style.

Highlights:
 - openings are SEeded from the user's own corpus (first K plies of random
   games) so the arena sees human-like middlegame starts
 - Stockfish strength is randomized per game (UCI_Elo range)
 - azuStyle temperature slightly randomized per game
 - PGN headers set White/Black to "AzuPre"/"Stockfish-<elo>" so
   prepare_dataset.py --accounts AzuPre picks the bot's moves.

Example:
  .venv/bin/python code/selfplay_arena.py --games 300 --out data/chesscom/arena.pgn \
      --concurrency 3
"""
import argparse
import chess
import chess.engine
import chess.pgn
import os
import random
import subprocess
import threading
import time

DEFAULT_SF = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", ".tools", "stockfish", "stockfish"))
DEFAULT_CKPT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "finetunes", "maia3-79m-v2", "best.pt"))
DEFAULT_CORPUS = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "data", "chesscom", "rapid_dataset_ALL.pgn"))

_reg, _reg_lock = [], threading.Lock()
_tl = threading.local()


def _register(eng):
    with _reg_lock:
        _reg.append(eng)


def close_all():
    with _reg_lock:
        for e in _reg:
            try:
                e.quit()
            except Exception:
                pass
        _reg.clear()


def get_engines(sf_path, ckpt, model):
    if getattr(_tl, "pair", None) is None:
        sf = chess.engine.SimpleEngine.popen_uci([sf_path], stderr=subprocess.DEVNULL, setpgrp=True)
        _register(sf)
        maia = chess.engine.SimpleEngine.popen_uci(
            [".venv/bin/maia3-uci", "--model", model,
             "--checkpoint-path", ckpt, "--use-uci-history"],
            stderr=subprocess.DEVNULL, setpgrp=True)
        _register(maia)
        _tl.pair = (sf, maia)
    return _tl.pair


def draw_opening_seed(corpus, max_plies=14):
    """Random game from user corpus → its first K plies as a starting position."""
    # chỉetermined: đọc ngẫu nhiên bằng cách skip N games (cheap: reuse while-loop)
    with open(corpus, "r", encoding="utf-8", errors="replace") as f:
        # skip a random prefix of games
        skip = random.randint(0, 1500)
        for _ in range(skip):
            g = chess.pgn.read_game(f)
            if g is None:
                break
        g = chess.pgn.read_game(f)
    if g is None:
        return []
    moves = list(g.mainline_moves())
    k = random.randint(2, max_plies)
    return moves[:min(k, len(moves))]


def play_one(i, sf_path, model, ckpt, corpus, elo_min, elo_max, azu_moves_max=200):
    sf, maia = get_engines(sf_path, ckpt, model)

    elo = random.randint(elo_min, elo_max)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1, "Hash": 64})
    temp = random.choice([0.2, 0.25, 0.3, 0.35, 0.4])
    maia.configure({"Temperature": str(temp), "Elo": random.choice([1460, 1550, 1700, 1800])})

    opening = draw_opening_seed(corpus)
    board = chess.Board()
    azu_color = chess.WHITE if i % 2 == 0 else chess.BLACK
    # seed moves: whoever's turn in the seed just plays those moves (both engines "replay")
    for mv in opening:
        board.push(mv)

    history_pgn_game = chess.pgn.Game()
    node = history_pgn_game
    for mv in opening:
        node = node.add_main_variation(mv)

    for j in range(azu_moves_max):
        if board.is_game_over():
            break
        if board.turn == azu_color:
            r = maia.play(board, chess.engine.Limit(time=0.35))
        else:
            r = sf.play(board, chess.engine.Limit(time=0.2))
        board.push(r.move)
        node = node.add_main_variation(r.move)

    result = board.result(claim_draw=True)
    if not board.is_game_over(claim_draw=True):
        # adjudicate by material/eval as a draw-ish finish if exceeded cap
        result = "1/2-1/2" if abs(count_material(board)) < 2 else ("1-0" if is_white_up(board) else "0-1")

    game = history_pgn_game
    w = "AzuPre" if azu_color == chess.WHITE else f"Stockfish-{elo}"
    b = f"Stockfish-{elo}" if azu_color == chess.WHITE else "AzuPre"
    game.headers["White"] = w
    game.headers["Black"] = b
    game.headers["Result"] = result
    game.headers["WhiteElo"] = "1700" if w == "AzuPre" else str(elo)
    game.headers["BlackElo"] = str(elo) if b.startswith("Stockfish") else "1700"
    game.headers["Event"] = "arena selfplay vs limited-stockfish"
    game.headers["Site"] = f"local arena#{i}"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["Termination"] = f"arena; sf_elo={elo}; azu_temp={temp}"
    return game


def count_material(board):
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    s = 0
    for sq, pc in board.piece_map().items():
        sign = 1 if pc.color == chess.WHITE else -1
        s += sign * vals.get(pc.piece_type, 0)
    return s


def is_white_up(board):
    return count_material(board) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--out", default="data/chesscom/arena.pgn")
    ap.add_argument("--model", default="maia3-79m")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--sf", default=DEFAULT_SF)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS, help="PGN để seed khai cuộc (corpus của bạn)")
    ap.add_argument("--elo-min", type=int, default=1500)
    ap.add_argument("--elo-max", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=3, help="Số ván song song (mỗi ván 1 GPU engine + 1 SF)")
    ap.add_argument("--max-plies", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"arena: {args.games} games vs Stockfish elo {args.elo_min}-{args.elo_max}, "
          f"concurrency {args.concurrency}", flush=True)

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
    n_done = 0
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        with open(args.out, "w") as out:
            futs = [pool.submit(play_one, i, args.sf, args.model, args.checkpoint,
                                args.corpus, args.elo_min, args.elo_max, args.max_plies)
                    for i in range(args.games)]
            for fut in as_completed(futs):
                g = fut.result()
                with lock:
                    out.write(str(g) + "\n\n"); out.flush()
                    results[g.headers.get("Result", "*")] = results.get(g.headers.get("Result", "*"), 0) + 1
                n_done += 1
                if n_done % 25 == 0:
                    rate = n_done / (time.time() - t0) * 60
                    with lock:
                        pending = results.copy()
                    print(f"  {n_done}/{args.games} ({rate:.1f} ván/phút) score {pending}", flush=True)

    close_all()
    dt = time.time() - t0
    print(f"\nDONE {n_done} games in {dt/60:.1f} min | kết quả: {results}")
    print(f"PGN -> {args.out}")
    print("→ để dựng dataset: .venv/bin/python code/prepare_dataset.py data/chesscom/arena.pgn "
          "--accounts AzuPre --out data/datasets/arena")


if __name__ == "__main__":
    main()

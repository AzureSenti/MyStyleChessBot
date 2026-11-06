#!/usr/bin/env python3
"""
Match: engine A (checkpoint A) vs engine B (checkpoint B), same Elo settings,
opening seeds drawn from the user's corpus, colors alternate each game.

Usage:
  .venv/bin/python code/engine_match.py \
      --engine-a finetunes/maia3-79m/best.pt --engine-b finetunes/maia3-79m-v2/best.pt \
      --games 50 --elo 1700 --temp 0.36 --out data/chesscom/match_v1_vs_v2.pgn
"""
import argparse
import chess
import chess.engine
import chess.pgn
import os
import random
import statistics
import subprocess
import threading
import time

MAIA_BIN = os.environ.get("MAIA_BIN", ".venv/bin/maia3-uci")
DEFAULT_SF_CORPUS = os.path.normpath(os.path.join(
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


def get_pair(ckpt_a, ckpt_b, elo, temp):
    """Mỗi thread có riêng 1 cặp (engineA, engineB)."""
    if getattr(_tl, "pair", None) is None:
        engA = spawn(ckpt_a, elo, temp)
        engB = spawn(ckpt_b, elo, temp)
        _tl.pair = (engA, engB)
    return _tl.pair


def spawn(ckpt, elo, temp):
    eng = chess.engine.SimpleEngine.popen_uci(
        [MAIA_BIN, "--model", "maia3-79m", "--checkpoint-path", ckpt,
         "--use-uci-history", "--elo", str(elo), "--temperature", str(temp)],
        stderr=subprocess.DEVNULL, setpgrp=True)
    _register(eng)
    return eng


def draw_opening_seed(corpus, max_plies=14):
    with open(corpus, "r", encoding="utf-8", errors="replace") as f:
        skip = random.randint(0, 1600)
        g = None
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


def material(board):
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    s = 0
    for pc in board.piece_map().values():
        s += (1 if pc.color == chess.WHITE else -1) * vals.get(pc.piece_type, 0)
    return s


def play_one(i, engA, engB, corpus, max_plies):
    opening = draw_opening_seed(corpus)
    node = g = chess.pgn.Game()
    for mv in opening:
        node = node.add_main_variation(mv)
    board = chess.Board()
    for mv in opening:
        board.push(mv)
    a_is_white = i % 2 == 0
    while not board.is_game_over(claim_draw=False) and board.ply() < max_plies:
        if board.turn == chess.WHITE:
            r = (engA if a_is_white else engB).play(board, chess.engine.Limit(time=0.35))
        else:
            r = (engB if a_is_white else engA).play(board, chess.engine.Limit(time=0.35))
        board.push(r.move)
        node = node.add_main_variation(r.move)
    result = board.result(claim_draw=True)
    if not board.is_game_over(claim_draw=True) and board.ply() >= max_plies:
        result = "1/2-1/2"
    g.headers["White"] = "A" if a_is_white else "B"
    g.headers["Black"] = "B" if a_is_white else "A"
    g.headers["Result"] = result
    g.headers["Event"] = "engine match"
    g.headers["Site"] = f"match#{i}"
    g.headers["Date"] = time.strftime("%Y.%m.%d")
    return g


def run_match(label, ckpt_a, ckpt_b, args, stats_all):
    print(f"\n{'='*66}\nMATCH {label}\n  A = {ckpt_a}\n  B = {ckpt_b}\n"
          f"  {args.games} ván · Elo {args.elo} · temp {args.temp}\n{'='*66}", flush=True)
    reg = {"wins": 0, "draws": 0, "losses": 0, "as_white": [0, 0, 0], "as_black": [0, 0, 0]}
    t0 = time.time()
    pgns = []
    pool = None
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = []
        for i in range(args.games):
            def task(i=i):
                engA, engB = get_pair(ckpt_a, ckpt_b, args.elo, args.temp)
                return play_one(i, engA, engB, args.corpus, args.max_plies)
            futs.append(pool.submit(task))
        for fut in as_completed(futs):
            try:
                g = fut.result()
            except Exception as exc:
                print(f"  ! 1 game failed: {exc!r} — bỏ qua", flush=True)
                continue
            pgns.append(g)
            done = len(pgns)
            if done % 10 == 0:
                print(f"  ...{done}/{args.games} ván ({time.time()-t0:.0f}s)", flush=True)
            res = g.headers["Result"]
            a_is_white = g.headers["White"] == "A"
            a_win = (res == "1-0" and a_is_white) or (res == "0-1" and not a_is_white)
            draw = res == "1/2-1/2"
            if a_win:
                reg["wins"] += 1
                key = "as_white" if a_is_white else "as_black"
                reg[key][0] += 1
            elif draw:
                reg["draws"] += 1
                reg[("as_white" if a_is_white else "as_black")][1] += 1
            else:
                reg["losses"] += 1
                reg[("as_white" if a_is_white else "as_black")][2] += 1
    n = max(1, reg["wins"] + reg["draws"] + reg["losses"])
    print(f"\nKẾT QUẢ {label} (góc nhìn A = {os.path.basename(os.path.dirname(ckpt_a))}):")
    print(f"  A thắng {reg['wins']}/{n} ({100*reg['wins']/n:.0f}%) · "
          f"hòa {reg['draws']} · A thua {reg['losses']} ({100*reg['losses']/n:.0f}%)")
    for side, name in (("as_white", "A làm Trắng"), ("as_black", "A làm Đen")):
        w, d, l = reg[side]
        print(f"  {name}: {w}W / {d}D / {l}L")
    stats_all[label] = reg
    if args.out:
        path = f"{os.path.splitext(args.out)[0]}_{label.replace(' ', '_').replace('->','_vs_').lower()}.pgn"
        with open(path, "w") as f:
            for g in pgns:
                print(g, file=f, end="\n\n")
        print(f"  PGN -> {path}")
    close_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--engine-b", required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--elo", type=int, default=1700)
    ap.add_argument("--temp", type=float, default=0.36)
    ap.add_argument("--corpus", default=DEFAULT_SF_CORPUS)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(7)
    stats = {}
    run_match("A vs B", args.engine_a, args.engine_b, args, stats)
    close_all()
    print("\nTỔNG:", stats)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Filter games by quality using Stockfish accuracy analysis.

For every game in a PGN corpus, this measures how well YOUR moves played:
  - centipawn loss (cpl) per your move = best-available eval − eval-after-your-move
  - blunders (your moves losing >= --blunder-cpl centipawns)
  - lichess-style accuracy % (win% model, per move)

Games that pass the thresholds are written to a filtered PGN, ready to feed
prepare_dataset.py. A CSV with per-game metrics is produced for inspection.

Example:
  .venv/bin/python code/filter_games.py \
      --pgn data/chesscom/rapid_dataset_ALL.pgn \
      --accounts TsukinoLuna Sasayaku \
      --out-pgn  data/chesscom/rapid_dataset_filtered.pgn \
      --out-csv  data/chesscom/game_quality.csv \
      --workers 12
"""

import argparse
import chess
import chess.engine
import chess.pgn
import csv
import math
import os
import statistics
import subprocess
import threading
import time

DEFAULT_SF = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", ".tools", "stockfish", "stockfish"))

_g = threading.local()  # one Stockfish per worker thread


def get_engine(sf_path):
    eng = getattr(_g, "engine", None)
    if eng is None:
        eng = chess.engine.SimpleEngine.popen_uci([sf_path],
                                                  stderr=subprocess.DEVNULL, setpgrp=True)
        eng.configure({"Threads": 1, "Hash": 64})
        _g.engine = eng
        with _reg_lock:
            _registry.append(eng)
    return eng


_registry, _reg_lock = [], threading.Lock()


def close_all_engines():
    with _reg_lock:
        for eng in _registry:
            try:
                eng.close()
            except Exception:
                pass
        _registry.clear()


def cp(score, cap=10000):
    """engine score (relative = side-to-move) → centipawns, mate mapped to ±cap"""
    if score.is_mate():
        return cap if score.mate() > 0 else -cap
    return score.score()


def win_percent(cp_):
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp_)) - 1)


def move_accuracy(cp_best, cp_played):
    """lichess accuracy formula applied to a single move (user perspective)."""
    wp_diff = win_percent(cp_best) - win_percent(cp_played)
    a = 103.1668 * math.exp(-0.04354 * max(0.0, wp_diff)) - 3.1669
    return max(0.0, min(100.0, a))


def analyze_game(game, accounts, sf_path, depth, blunder_cpl):
    """Return dict with per-account metrics for this game, or None if irrelevant."""
    board = game.board()
    colors = {}
    wanted = {a.lower(): a for a in accounts}
    white = game.headers.get("White", "").lower()
    black = game.headers.get("Black", "").lower()
    if white in wanted:
        colors[chess.WHITE] = wanted[white]
    if black in wanted:
        colors[chess.BLACK] = wanted[black]
    if not colors:
        return None

    eng = get_engine(sf_path)
    limit = chess.engine.Limit(depth=depth)

    moves_by = {c: 0 for c in colors}
    plies_of = {c: [] for c in colors}  # (cpl, cp_best, cp_played)

    node_board = board
    for move in game.mainline_moves():
        mover = node_board.turn
        if mover in colors:
            try:
                s_best = eng.analyse(node_board, limit, info=chess.engine.INFO_SCORE)["score"].relative
                child = node_board.copy(stack=False)
                child.push(move)
                s_played = eng.analyse(child, limit, info=chess.engine.INFO_SCORE)["score"].relative
            except chess.engine.EngineError:
                node_board.push(move)
                continue
            # played move's eval from the MOVER's perspective = -score(child, side-to-mover=opponent)
            cp_best = cp(s_best)
            cp_played = -cp(s_played)
            cpl = max(0, min(cp_best - cp_played, 1000))
            plies_of[mover].append((cpl, cp_best, cp_played))
            moves_by[mover] += 1
        node_board.push(move)

    result = {
        "game_id": game_id_of(game),
        "white": game.headers.get("White", ""),
        "black": game.headers.get("Black", ""),
        "result": game.headers.get("Result", ""),
        "termination": game.headers.get("Termination", ""),
        "plies": node_board.ply(),
        "datetime": game.headers.get("UTCDate", "") + " " + game.headers.get("UTCTime", ""),
    }
    ok = False
    for color, name in colors.items():
        n = moves_by[color]
        if n == 0:
            continue
        rows = plies_of[color]
        avg_cpl = statistics.fmean(r[0] for r in rows)
        accuracy = statistics.fmean(move_accuracy(r[1], r[2]) for r in rows)
        blunders = sum(1 for r in rows if r[0] >= blunder_cpl)
        result[name] = {
            "user_moves": n,
            "avg_cpl": round(avg_cpl, 1),
            "accuracy": round(accuracy, 2),
            "blunders": blunders,
            "blunder_rate": round(blunders / n, 3),
        }
        ok = True

    return result if ok else None


def game_id_of(game):
    """Chess.com games carry the numeric id in the Link header."""
    for key in ("Link", "ChesscomID", "Site"):
        v = game.headers.get(key, "")
        if v and "/" in v:
            return v.rstrip("/").split("/")[-1]
    return v or "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--accounts", nargs="+", required=True, help="users whose moves are judged")
    ap.add_argument("--out-pgn", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--sf", default=DEFAULT_SF, help="Stockfish binary")
    ap.add_argument("--depth", type=int, default=10, help="Stockfish depth per position eval")
    ap.add_argument("--blunder-cpl", type=int, default=100, help="cpl that counts as a blunder")
    ap.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 4) - 1)))
    # quality gate (accuracy门槛 phân theo kết quả ván như user muốn)
    ap.add_argument("--min-accuracy-win", type=float, default=55.0,
                    help="accuracy tối thiểu cho vÁN THẮNG")
    ap.add_argument("--min-accuracy-draw", type=float, default=55.0,
                    help="accuracy tối thiểu cho ván HÒA")
    ap.add_argument("--min-accuracy-loss", type=float, default=60.0,
                    help="accuracy tối thiểu cho ván THUA")
    ap.add_argument("--max-avg-cpl", type=float, default=None,
                    help="optional hard cap on avg centipawn loss (None = tắt)")
    ap.add_argument("--max-blunder-rate", type=float, default=None,
                    help="optional cap on blunder share (None = tắt)")
    ap.add_argument("--min-user-moves", type=int, default=10, help="skip tiny/aborted games")
    args = ap.parse_args()

    if not os.path.exists(args.sf):
        sys_exit = SystemExit(f"Stockfish not found at {args.sf}")
        raise sys_exit

    os.makedirs(os.path.dirname(os.path.abspath(args.out_pgn)), exist_ok=True)

    games = []
    with open(args.pgn, "r", encoding="utf-8", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            n_ply = sum(1 for _ in g.mainline_moves())
            if n_ply >= 2 * args.min_user_moves:
                games.append(g)
    print(f"loaded {len(games)} games (>= {2*args.min_user_moves} plies)", flush=True)

    results, done = [], 0
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(analyze_game, g, args.accounts, args.sf, args.depth,
                            args.blunder_cpl): g for g in games}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                results.append(r)
            done += 1
            if done % 150 == 0:
                print(f"  ... {done}/{len(games)} analyzed ({time.time()-t0:.0f}s)", flush=True)

    # pick THE account played for the keep decision (self-play between accounts is
    # vanishingly rare; if it happens the weaker side's metrics decide)
    for r in results:
        pairs = [(a, r[a]) for a in args.accounts if a in r]
        acc_name, best = pairs[0] if pairs else ("", {})
        r["_account"] = acc_name
        r["_acc"] = best.get("accuracy", 0)
        r["_cpl"] = best.get("avg_cpl", 9999) if best.get("avg_cpl") is not None else 9999
        r["_brate"] = best.get("blunder_rate", 1) if best.get("blunder_rate") is not None else 1
        r["_moves"] = best.get("user_moves", 0)

    def passes(r):
        # kết quả ván từ góc nhìn của account được train
        acc_name = r.get("_account", "").lower()
        is_white = r["white"].lower() == acc_name
        res = r["result"]
        if res == "1/2-1/2":
            outcome, thr = "draw", args.min_accuracy_draw
        elif (is_white and res == "1-0") or (not is_white and res == "0-1"):
            outcome, thr = "win", args.min_accuracy_win
        else:
            outcome, thr = "loss", args.min_accuracy_loss
        r["_outcome"] = outcome
        if r["_acc"] < thr:
            return False
        if args.max_avg_cpl is not None and r["_cpl"] > args.max_avg_cpl:
            return False
        if args.max_blunder_rate is not None and r["_brate"] > args.max_blunder_rate:
            return False
        return r["_moves"] >= args.min_user_moves

    kept_ids = {r["game_id"] for r in results if passes(r)}
    rejected = [r for r in results if not passes(r)]
    accs = sorted(r["_acc"] for r in results)

    # write CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game_id", "account", "outcome", "user_moves", "accuracy", "avg_cpl",
                    "blunders", "blunder_rate", "result", "termination", "datetime",
                    "white", "black", "kept"])
        for r in sorted(results, key=lambda x: -x["_acc"]):
            for a in args.accounts:
                if a in r:
                    u = r[a]
                    w.writerow([r["game_id"], a, r.get("_outcome", "?"), u["user_moves"],
                                u["accuracy"], u["avg_cpl"], u["blunders"], u["blunder_rate"],
                                r["result"], r["termination"], r["datetime"],
                                r["white"], r["black"], 1 if r["game_id"] in kept_ids else 0])

    # stream filtered PGN
    n_written = 0
    with open(args.pgn, "r", encoding="utf-8", errors="replace") as f, \
            open(args.out_pgn, "w") as out:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            gid = game_id_of(g)
            if gid in kept_ids:
                print(g, file=out, end="\n\n")
                n_written += 1

    close_all_engines()
    n = max(1, len(accs))
    print(f"\ndone in {time.time()-t0:.0f}s — analyzed {len(results)} games "
          f"involving {', '.join(args.accounts)}")
    print(f"accuracy distribution: p10={accs[n//10]:.0f}  p50={accs[n//2]:.0f}  p90={accs[9*n//10]:.0f}")
    print(f"KEPT {len(kept_ids):5d} games  (acc: win>={args.min_accuracy_win}, "
          f"draw>={args.min_accuracy_draw}, loss>={args.min_accuracy_loss})")
    print(f"DROP {len(rejected):5d} games")
    print(f"\nfiltered PGN -> {args.out_pgn} ({n_written} games)")
    print(f"CSV          -> {args.out_csv}")


if __name__ == "__main__":
    main()

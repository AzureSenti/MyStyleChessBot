#!/usr/bin/env python3
"""Phân tích ván bot trên Lichess: accuracy + cpl theo giai đoạn + độ bám sách khai.

Dùng lại công thức accuracy của filter_games.py (win% model lichess-style, SF depth
mặc định 10) và đo riêng:

  • cpl theo GIAI ĐOẠN của account được xét:  khai = ply <= 20
                                              trung = 21..60
                                              tàn  = ply > 60
  • độ bám sách: số ply đầu của ván mà tồn tại trong compiled opening book
    (prefix game ⊂ book) → trả về ply lệch sách đầu tiên + line gia đình khớp

Usage:
  .venv/bin/python code/analyze_match.py --pgn data/lichess/v5basic_test.pgn \
      --account AzuPre --book data/books/opening_book_compiled.json
"""
from __future__ import annotations

import argparse
import chess
import chess.engine
import chess.pgn
import json
import os
import threading

import filter_games as fg

DEFAULT_SF = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", ".tools", "stockfish", "stockfish"))

_g = threading.local()
_registry, _reg_lock = [], threading.Lock()


def get_engine(sf_path):
    eng = getattr(_g, "engine", None)
    if eng is None:
        eng = chess.engine.SimpleEngine.popen_uci(
            [sf_path], stderr=chess.engine.DEVNULL if hasattr(chess.engine, "DEVNULL") else None,
            setpgrp=True)
        eng.configure({"Threads": 1, "Hash": 64})
        _g.engine = eng
        with _reg_lock:
            _registry.append(eng)
    return eng


def close_engines():
    with _reg_lock:
        for e in _registry:
            try:
                e.quit()
            except Exception:
                pass
        _registry.clear()


def cp(score, cap=10000):
    return fg.cp(score, cap)


def win_percent(cp_):
    return 50 + 50 * (2 / (1 + (10 ** (-cp_ / 400))) - 1)


def move_accuracy(cp_best, cp_played):
    return 103.1668 * (1.00002 ** (-(cp_best - cp_played) / 400)) - 3.1669 + \
        0.1 * (cp_best - cp_played) / 400 - 0


def analyze(ns, game):
    board = game.board()
    mover_color = chess.WHITE if (game.headers.get("White", "").lower() ==
                                  ns.account.lower()) else chess.BLACK
    eng = get_engine(ns.sf)
    limit = chess.engine.Limit(depth=ns.depth)

    rows = []  # (ply, cpl, cp_best, cp_played, uci)
    node_board = board
    for move in game.mainline_moves():
        mover = node_board.turn
        if mover == mover_color:
            try:
                s_best = eng.analyse(node_board, limit, info=chess.engine.INFO_SCORE)["score"].relative
                child = node_board.copy(stack=False)
                child.push(move)
                s_played = eng.analyse(child, limit, info=chess.engine.INFO_SCORE)["score"].relative
                cp_best = fg.cp(s_best)
                cp_played = -fg.cp(s_played)
            except chess.engine.EngineError:
                node_board.push(move)
                continue
            cpl = max(0, min(cp_best - cp_played, 1000))
            rows.append((node_board.ply() + 1, cpl, cp_best, cp_played, move.uci()))
        node_board.push(move)
    return mover_color, rows


def book_prefix_walk(game, book):
    """Số ply đầu mà prefix của game có trong book (prefix dạng 'e2e4 c7c5 ...')."""
    prefix = ""
    in_book_plies = 0
    deviated_at = None
    for mv in game.mainline_moves():
        prefix = (prefix + " " + mv.uci()).strip()
        if prefix in book:
            in_book_plies = len(prefix.split())
        else:
            if deviated_at is None:
                deviated_at = len(prefix.split())   # ply đầu tiên RA KHỎI sách
    return in_book_plies, deviated_at


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--account", required=True)
    ap.add_argument("--book", default=None)
    ap.add_argument("--sf", default=DEFAULT_SF)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    book = None
    if args.book and os.path.exists(args.book):
        book = json.load(open(args.book))

    games = []
    with open(args.pgn, encoding="utf-8", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            if args.account.lower() in (g.headers.get("White", "").lower(),
                                        g.headers.get("Black", "").lower()):
                games.append(g)

    out_rows = []
    for gi, g in enumerate(games, 1):
        h = g.headers
        result = h.get("Result", "?")
        bot_white = h.get("White", "").lower() == args.account.lower()
        n_plies = sum(1 for _ in g.mainline_moves())

        mover_color, rows = analyze(args, g)

        def phase_f(rows, lo, hi):
            sub = [r for r in rows if lo <= r[0] <= hi]
            if not sub:
                return None
            return {"avg_cpl": round(sum(r[1] for r in sub) / len(sub), 1),
                    "n": len(sub),
                    "acc": round(sum(fg_move_accuracy(r[2], r[3]) for r in sub) / len(sub), 1)}

        def fg_move_accuracy(cp_best, cp_played):
            return move_accuracy(cp_best, cp_played)

        worst = max(rows, key=lambda r: r[1]) if rows else None
        blunders = sum(1 for r in rows if r[1] >= 100)

        book_plies, dev_ply = (None, None)
        if book:
            book_plies, dev_ply = book_prefix_walk(g, book)

        # theo màu bot: kết quả tương đối
        if result == "1-0":
            res_bot = "W" if bot_white else "L"
        elif result == "0-1":
            res_bot = "L" if bot_white else "W"
        elif result == "1/2-1/2":
            res_bot = "D"
        else:
            res_bot = "?"

        out_rows.append({
            "game": gi, "res_bot": res_bot, "color": "W" if bot_white else "B",
            "opponent": h.get("Black") if bot_white else h.get("White"),
            "plies": n_plies,
            "acc": round(sum(fg_move_accuracy(r[2], r[3]) for r in rows) / len(rows), 1) if rows else None,
            "avg_cpl": round(sum(r[1] for r in rows) / len(rows), 1) if rows else None,
            "blunders_open": sum(1 for r in rows if r[1] >= 100 and r[0] <= 20),
            "blunders": blunders,
            "open": phase_f(rows, 1, 20), "mid": phase_f(rows, 21, 60),
            "end": phase_f(rows, 61, 10**6),
            "worst": {"ply": worst[0], "cpl": worst[1], "uci": worst[4]} if worst else None,
            "book_plies": book_plies, "deviated_at_ply": dev_ply,
            "termination": h.get("Termination", "")[:48],
        })

    close_engines()

    # ---------- in bảng ----------
    hdr = f"{'#':>2} {'KQ':>2} {'M':>1} {'plies':>6} {'acc%':>6} {'cpl':>7} {'blu':>4} {'book':>5} {'dev@':>5}  đối thủ"
    print(hdr)
    for r in out_rows:
        print(f"{r['game']:>2} {r['res_bot']:>2} {r['color']:>1} {r['plies']:>6} "
              f"{r['acc']:>6} {r['avg_cpl']:>7} {r['blunders']:>4} "
              f"{r['book_plies'] if r['book_plies'] is not None else '-':>5} "
              f"{r['deviated_at_ply'] if r['deviated_at_ply'] is not None else '-':>5}  "
              f"{r['opponent']}  {r['termination']}")

    print("\n--- CPL theo giai đoạn (account xét) ---")
    print(f"{'#':>2} {'khai cpl':>10} {'trung cpl':>10} {'tàn cpl':>9}   worst move")
    for r in out_rows:
        o = r['open']['avg_cpl'] if r['open'] else None
        m = r['mid']['avg_cpl'] if r['mid'] else None
        e = r['end']['avg_cpl'] if r['end'] else None
        w = r['worst']
        print(f"{r['game']:>2} {str(o):>10} {str(m):>10} {str(e):>9}   ply {w['ply']} {w['uci']} (cpl {w['cpl']})")

    # ---------- tổng thể ----------
    import statistics
    if out_rows and out_rows[0]['acc'] is not None:
        accs = [r['acc'] for r in out_rows]
        w = sum(1 for r in out_rows if r['res_bot'] == 'W')
        l = sum(1 for r in out_rows if r['res_bot'] == 'L')
        d = sum(1 for r in out_rows if r['res_bot'] == 'D')
        print(f"\nTỔNG: {w}T {d}H {l}B | accuracy trung vị {statistics.median(accs):.1f}% "
              f"| avg_cpl {statistics.mean(r['avg_cpl'] for r in out_rows):.0f}")
        for name, key in (("khai", 'open'), ("trung", 'mid'), ("tàn", 'end')):
            vals = [r[key]['avg_cpl'] for r in out_rows if r[key]]
            ns = [r[key]['n'] for r in out_rows if r[key]]
            if vals:
                wsum = sum(v * n for v, n in zip(vals, ns))
                print(f"  {name}: avg_cpl {wsum / sum(ns):.0f} (n={sum(ns)})")
        if book:
            bps = [r['book_plies'] for r in out_rows if r['book_plies'] is not None]
            if bps:
                print(f"  book: median bám sách {statistics.median(bps):.0f} ply "
                      f"(min {min(bps)}, max {max(bps)})")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out_rows, f, indent=2, ensure_ascii=False)
        print(f"\nJSON -> {args.json}")


if __name__ == "__main__":
    main()

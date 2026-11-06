#!/usr/bin/env python3
import math, statistics, subprocess
import chess, chess.engine, chess.pgn

SF = ".tools/stockfish/stockfish"

def cp_of(score, cap=10000):
    if score.is_mate():
        return cap if score.mate() > 0 else -cap
    return score.score()

def winp(c):
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * c)) - 1)

def move_acc(cp_best, cp_played):
    wpb, wpa = winp(cp_best), winp(cp_played)
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * max(0.0, wpb - wpa)) - 3.1669))

def fmt_eval(ev):
    if ev > 9000:
        return "M" if ev > 0 else "-M"
    return f"{ev/100:+.2f}"

def analyze(path, label):
    g = chess.pgn.read_game(open(path))
    h = g.headers
    w, b = h["White"], h["Black"]
    print("=" * 72)
    print(f"{label}: {w}({h['WhiteElo']}) vs {b}({h['BlackElo']}) — {h['Result']}")
    print(f"Khai cuộc: {h.get('Opening','?')} | Kết thúc: {h.get('Termination','')}")
    print(f"Link: {h['Site']}")
    print("-" * 72)

    eng = chess.engine.SimpleEngine.popen_uci([SF], stderr=subprocess.DEVNULL, setpgrp=True)
    eng.configure({"Threads": 2, "Hash": 256})
    limit = chess.engine.Limit(depth=12)

    board = g.board()
    plies = list(g.mainline_moves())
    stats = {w: {"cp_pairs": [], "lossmoves": []}, b: {"cp_pairs": [], "lossmoves": []}}
    evals_white = []  # eval sau từng nước, góc nhìn TRẮNG nhất quán

    for i, move in enumerate(plies):
        mover = board.turn
        mover_name = w if mover == chess.WHITE else b
        s_best = eng.analyse(board, limit)["score"].relative
        child = board.copy(stack=False)
        child.push(move)
        s_played = eng.analyse(child, limit)["score"].relative
        cp_best, cp_played = cp_of(s_best), -cp_of(s_played)  # góc mover
        cpl = max(0, min(cp_best - cp_played, 1000))
        stats[mover_name]["cp_pairs"].append((cpl, cp_best, cp_played))
        if cpl >= 150:
            san = board.copy(stack=False).variation_san([move])
            stats[mover_name]["lossmoves"].append((i + 1, san, cpl))
        evalw = cp_played if mover == chess.WHITE else -cp_played
        evals_white.append(evalw)
        board.push(move)

    for role, s in stats.items():
        pairs = s["cp_pairs"]
        acc = statistics.fmean(move_acc(p[1], p[2]) for p in pairs)
        avg_cpl = statistics.fmean(p[0] for p in pairs)
        nb = sum(1 for p in pairs if p[0] >= 150)
        print(f"\n{role}: {len(pairs)} nước | accuracy {acc:.1f} | avg_cpl {avg_cpl:.0f} | nước tệ ≥150cp: {nb}")
        for ply, san, loss in sorted(s["lossmoves"], key=lambda x: -x[2])[:5]:
            print(f"    nước #{ply:3d}: {san:22s} làm mất {loss:5.0f} cp")

    print("\nKHOẢNH KHẮC CỜ ĐỔI CHIỀU LỚN NHẤT (bỏ pha chiếu hết cuối game):")
    swings = []
    for i in range(1, len(evals_white)):
        d = evals_white[i] - evals_white[i - 1]
        if abs(d) >= 150 and abs(evals_white[i]) < 9000:
            mover_name = w if i % 2 == 1 else b
            swings.append((abs(d), i + 1, mover_name, d, evals_white[i]))
    for mag, ply, name, d, ev in sorted(swings, key=lambda x: -x[0])[:12]:
        advant = "Trắng" if ev > 0 else "Đen"
        print(f"    nước #{ply:3d} ({name:9s} đi): {abs(d):5.0f} cp về lợi {advant} — vị thế {fmt_eval(ev)}")
    if not swings:
        print("    (không có swing ≥150cp ngoài tàn cuộc)")

    eng.quit()

import sys
if __name__ == "__main__":
    analyze(sys.argv[1], "VÁN: " + sys.argv[2])
    analyze()

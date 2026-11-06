#!/usr/bin/env python3
"""Hồ sơ phong cách cờ — quét corpus, đo từng nước của account bằng SF,
tổng hợp thành báo cáo phong cách + điểm yếu + đề xuất luyện tập.

Cách dùng:
  .venv/bin/python code/style_report.py \
      --pgn data/chesscom/all_rapid_blitz.pgn \
      --accounts TsukinoLuna Sasayaku \
      --out-json docs/style_report.json

Nó đo gì (per account):
  • Tỷ số W/D/L tổng, theo MÀU, theo time control (rapid/blitz)
  • cpl & accuracy theo GIAI ĐOẠN (khai ≤20 · trung 21–60 · tàn >60)
  • blunder theo GIAI ĐOẠN và theo GIỜ ĐỒNG HỒ (<30s / 30–90 / >90)
  • Nước hỏng nhất mỗi ván → ply "sự cố đầu tiên" là bao nhiêu
  • Mate bế (SF bảo có mate mà mình chơi nước khác)
  • Chia thành (castle) sớm muộn: % ván castle + median ply
  • Chỉ số hung hãn: % nước là ăn quân / chiếu / đẩy quả tức thời
  • Quản lý giờ: median thời gian suy nghĩ từng giai đoạn, % nước <2s,
    % thua do chạy hết giờ
  • Top khai (4 ply đầu) với W% + cpl của mình → nhánh mở nào đang \\

Xuất: bảng tiếng Việt ra stdout + JSON đầy đủ (--out-json).
"""
from __future__ import annotations

import argparse
import chess
import chess.engine
import chess.pgn
import json
import os
import re
import statistics
import subprocess
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import filter_games as fg   # cp(), win_percent(), accuracy công thức lichess

DEFAULT_SF = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", ".tools", "stockfish", "stockfish"))

CLK_RE = re.compile(r"clk\s+(\d+):(\d+):(\d+)(?:\.(\d+))?")
_g = threading.local()
_registry, _reg_lock = [], threading.Lock()

PHASES = (("khai", 0, 20), ("trung", 21, 60), ("tan", 61, 10_000))


def get_engine(sf_path):
    eng = getattr(_g, "engine", None)
    if eng is None:
        eng = chess.engine.SimpleEngine.popen_uci(
            [sf_path], stderr=subprocess.DEVNULL, setpgrp=True)
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


def clk_seconds(comment):
    m = CLK_RE.search(comment or "")
    if not m:
        return None
    h, mn, s, frac = m.groups()
    sec = int(h) * 3600 + int(mn) * 60 + int(s)
    if frac:
        sec += float("0." + frac)
    return sec


def phase_of(ply):
    for name, lo, hi in PHASES:
        if lo <= ply <= hi:
            return name
    return "tan"


# ------------------------------------------------------------------ per-game
def analyze_game(g, account, sf_path, depth, blunder_cpl):
    """Trả về dict mọi chỉ số theo mức VÁN cho account."""
    h = g.headers
    white, black = h.get("White", ""), h.get("Black", "")
    if account.lower() not in (white.lower(), black.lower()):
        return None
    own_white = white.lower() == account.lower()
    result = h.get("Result", "*")
    if result == "1-0":
        res = "W" if own_white else "L"
    elif result == "0-1":
        res = "L" if own_white else "W"
    elif result == "1/2-1/2":
        res = "D"
    else:
        res = "?"

    tc = h.get("TimeControl", "")
    base = None
    if tc and not tc.startswith("1/"):
        try:
            base = float(tc.split("+")[0])
        except ValueError:
            pass
    klass = "rapid" if (base is not None and base >= 600) else (
        "blitz" if base is not None else "lien-tuc")

    term = (h.get("Termination", "") or "").lower()
    won_on_time = "time" in term and res == "W"
    lost_on_time = "time" in term and res == "L"
    resigned = ("resign" in term or "won on resignation" in term) and res == "L"

    board = g.board()
    mover_color = chess.WHITE if own_white else chess.BLACK
    eng = get_engine(sf_path)
    limit = chess.engine.Limit(depth=depth)

    rows = []                # (ply, cpl, cp_best, cp_played, spent_sec, phase, uci, is_blunder)
    first_mistake_ply = None
    missed_mate_cnt = 0
    prev_clk = {}
    base_t = base or 600.0
    n_capt, n_check, n_quiet = 0, 0, 0
    castle_ply = None
    piece_moves = Counter()

    node = g
    while node.variations:
        child = node.variations[0]
        move = child.move
        if move is None:
            break
        ply = board.ply() + 1
        mover = board.turn
        if mover != mover_color:
            board.push(move); node = child
            continue

        san = board.san(move)
        piece = san[0] if san[0] in "KQRBN" else "P"
        piece_moves[piece] += 1
        if "x" in san:
            n_capt += 1
        if "+" in san or "#" in san:
            n_check += 1
        else:
            n_quiet += 1
        if board.has_castling_rights(mover) and san in ("O-O", "O-O-O") and not castle_ply:
            castle_ply = ply

        # thời gian: clk sau nước − clk trước nước (comment trên node con)
        clk = clk_seconds(child.comment)
        last = prev_clk.get(mover, base_t)
        spent = (last - clk) if (clk is not None and last > 0) else None
        prev_clk[mover] = clk if clk is not None else last

        try:
            s_best = eng.analyse(board, limit, info=chess.engine.INFO_SCORE)["score"].relative
        except chess.engine.EngineError:
            board.push(move); node = child
            continue
        mate_avail = s_best.is_mate() and (s_best.mate() or 0) > 0

        child_b = board.copy(stack=False)
        child_b.push(move)
        try:
            s_played = eng.analyse(child_b, limit, info=chess.engine.INFO_SCORE)["score"].relative
        except chess.engine.EngineError:
            board.push(move); node = child
            continue
        cp_best = fg.cp(s_best)
        cp_played = -fg.cp(s_played)
        cpl = max(0, min(cp_best - cp_played, 1000))
        blun = cpl >= blunder_cpl

        if mate_avail and not (s_played.is_mate() and (s_played.mate() or 0) > 0):
            missed_mate_cnt += 1
            blun = True
        if blun and first_mistake_ply is None:
            first_mistake_ply = ply

        rows.append((ply, cpl, cp_best, cp_played, spent, phase_of(ply),
                     move.uci(), blun))
        board.push(move); node = child

    if not rows:
        return None
    return {
        "res": res, "tc": klass, "own_white": own_white,
        "self_elo": int(h.get("WhiteElo", 0) if own_white else h.get("BlackElo", 0)),
        "oppo_elo": int(h.get("BlackElo", 0) if own_white else h.get("WhiteElo", 0)),
        "lost_on_time": lost_on_time, "won_on_time": won_on_time, "resigned": resigned,
        "plies": board.ply(),
        "rows": rows,
        "first_mistake_ply": first_mistake_ply,
        "missed_mate": missed_mate_cnt,
        "castle_ply": castle_ply,
        "captures": n_capt, "checks": n_check, "n_moves": len(rows),
        "piece_moves": dict(piece_moves),
    }


# ------------------------------------------------------------- aggregation
def agg_account(name, gds, args):
    R = defaultdict(list)  # prépar aggregate per bucket
    simple = defaultdict(list)
    for g in gds:
        simple["tc"].append(g["tc"])
        for ph, _lo, _hi in PHASES:
            for r in g["rows"]:
                if r[5] == ph:
                    R[f"cpl_{ph}"].append(r[1])
                    if r[7]:
                        R[f"blun_{ph}"].append(r[0])
                    clk = r[4]
                    if clk is not None:
                        R[f"spent_{ph}"].append(clk)
                        bucket = "duoi30s" if clk < 30 else ("30-90s" if clk < 90 else "tren90s")
                        if r[7]:
                            R[f"blun_{ph}_{bucket}"].append(1)
                        R[f"n_{ph}_{bucket}"].append(1)
        simple["res"].append(g["res"])
        simple["color"].append("W" if g["own_white"] else "B")
        simple["self_elo"].append(g["self_elo"])
        simple["oppo_elo"].append(g["oppo_elo"])
        if g["lost_on_time"]:
            simple["lost_on_time"].append(1)
        if g["resigned"]:
            simple["resigned"].append(1)
        if g["castle_ply"]:
            simple["castle_ply"].append(g["castle_ply"])
        simple["missed_mate"].append(g["missed_mate"])
        simple["captures"].append(g["captures"] / max(g["n_moves"], 1))
        simple["checks"].append(g["checks"] / max(g["n_moves"], 1))
        n_cpl = [r[1] for r in g["rows"] if r[4] is not None and r[4] < 2]
        if n_cpl:
            simple["avg_cpl_spatial_2s"].append(sum(n_cpl) / len(n_cpl))
            simple["n_moves_2s"].append(len(n_cpl))
        if g["first_mistake_ply"]:
            simple["first_mistake_ply"].append(g["first_mistake_ply"])

    njl = {}
    W = sum(1 for r in simple["res"] if r == "W")
    L = sum(1 for r in simple["res"] if r == "L")
    D = sum(1 for r in simple["res"] if r == "D")
    out = {
        "account": name, "games": len(gds),
        "score": f"{W}T {D}H {L}B", "score_pct": round(100 * (W + D / 2) / max(len(gds), 1), 1),
        "by_color": {
            "W": _subgroup(gds, pred=lambda g: g["own_white"]),
            "B": _subgroup(gds, pred=lambda g: not g["own_white"]),
        },
        "by_tc": {tc: _subgroup(gds, pred=(lambda g: g["tc"] == tc))
                  for tc in ("rapid", "blitz")},
        "phase": {ph: _phase_stats(R, ph) for ph, _lo, _hi in PHASES},
        "castling": {
            "pct_games_castled": round(100 * len(simple["castle_ply"]) / len(gds), 1),
            "median_ply": int(statistics.median(simple["castle_ply"])) if simple["castle_ply"] else None,
        },
        "aggression": {
            "captures_per_100_moves": round(100 * statistics.mean(simple["captures"]), 1),
            "checks_per_100_moves": round(100 * statistics.mean(simple["checks"]), 1),
        },
        "time_control": {
            "median_movetime_khai": _med(R.get("spent_khai")),
            "median_movetime_trung": _med(R.get("spent_trung")),
            "median_movetime_tan": _med(R.get("spent_tan")),
            "lost_on_time": len(simple["lost_on_time"]),
            "resigned_losses": len(simple["resigned"]),
        },
        "missed_mate_games": sum(1 for m in simple["missed_mate"] if m > 0),
        "missed_mate_total": sum(simple["missed_mate"]),
        "first_mistake_median_ply": _med(simple["first_mistake_ply"]),
    }
    # % nước <2s riêng cho đúng
    n2s = sum(simple["n_moves_2s"])
    if n2s:
        w2 = statistics.mean(simple["avg_cpl_spatial_2s"])
        out["instant_moves"] = {"n_moves_under_2s": n2s, "avg_cpl_under_2s": round(w2, 1)}
    return out


def _med(vals):
    return round(statistics.median(vals)) if vals else None


def _phase_stats(R, ph):
    cpl = R.get(f"cpl_{ph}", [])
    out = {"n_moves": len(cpl), "avg_cpl": round(statistics.mean(cpl), 1) if cpl else None,
           "median_cpl": _med(cpl)}
    for bucket in ("duoi30s", "30-90s", "tren90s"):
        nall = R.get(f"n_{ph}_{bucket}", [])
        nb = R.get(f"blun_{ph}_{bucket}", [])
        if nall:
            out[f"blunder_rate_{bucket}"] = round(100 * len(nb) / len(nall), 1)
    return out


def _subgroup(gds, key=None, pred=None):
    """Ngữ: lọc ván do pred (hoặc theo key) → W/D/L str + score."""
    def keep(g):
        if pred:
            return pred(g)
        return bool(g[key])
    sub = [g for g in gds if keep(g)]
    if not sub:
        return {"n": 0}
    w = sum(1 for g in sub if g["res"] == "W")
    d = sum(1 for g in sub if g["res"] == "D")
    l = sum(1 for g in sub if g["res"] == "L")
    return {"n": len(sub), "fmt": f"{w}T {d}H {l}B",
            "score_pct": round(100 * (w + d / 2) / len(sub), 1)}


# ------------------------------------------------------------------ insights
def recommend(out_per_account):
    for A in out_per_account:
        tips = []
        ph = A.get("phase", {})
        tan, trung, khai = ph.get("tan", {}), ph.get("trung", {}), ph.get("khai", {})
        if tan.get("avg_cpl") and khai.get("avg_cpl") and tan["avg_cpl"] > 1.5 * khai["avg_cpl"]:
            tips.append("cpl TÀN cao hơn khai ~%.0f%% → luyện endgame cơ bản (đường vua, qua trận, quảng se chốt) đóng ~15 phút trình này mỗi buổi"
                        % (100 * tan["avg_cpl"] / max(khai["avg_cpl"], 1)))
        b_tan_low = tan.get("blunder_rate_duoi30s")
        if b_tan_low is not None and tan.get("blunder_rate_tren90s") is not None and \
                b_tan_low > 2.5 * max(tan["blunder_rate_tren90s"], 0.5):
            tips.append("blunder tàn khi giờ cạn (<30s) là %.1f%% gần 2.5× lúc giờ ĐẺ — luyện tàn nhanh: 10-15 ván endgame 1+2'."
                        % b_tan_low)
        if A["castling"]["median_ply"] and A["castling"]["median_ply"] > 30:
            tips.append("castle muộn (median ply %d) → mục tiêu rõ:PLY ~15-25; sớm hoàng thành giúp tránh mát manh tiên hàm" % A["castling"]["median_ply"])
        if A["time_control"]["lost_on_time"] >= 3:
            tips.append("%d ván hoà giờ — dài hơi luôn đặt 'checkpoint giờ' mỗi 10 nước lui/đi." % A["time_control"]["lost_on_time"])
        if A["missed_mate_total"] >= 5:
            tips.append("bo Sollt %d lần 'có mate nhưng bỏ' — раз ch CON: 10 phút mating-patterns mỗi discontinuity tối." % A["missed_mate_total"])
        if A["first_mistake_median_ply"]:
            tips.append("lớp phong cách: sự cố đầu tiên thường ở ply %d — chủ động chấm qu ước vị thế, tái kế hoach khi tới gần đấy." % A["first_mistake_median_ply"])
        A["recommendations"] = tips
    return out_per_account


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--accounts", nargs="+", required=True)
    ap.add_argument("--sf", default=DEFAULT_SF)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--blunder-cpl", type=int, default=100)
    ap.add_argument("--min-ply", type=int, default=10)
    ap.add_argument("--workers", type=int, default=min(10, max(1, (os.cpu_count() or 4) - 1)))
    ap.add_argument("--out-json", default="docs/style_report.json")
    ap.add_argument("--out-pergame-csv", default=None)
    args = ap.parse_args()

    t0 = time.time()
    games = []
    with open(args.pgn, encoding="utf-8", errors="replace") as f:
        while True:
            try:
                g = chess.pgn.read_game(f)
            except Exception:
                continue
            if g is None:
                break
            if g.headers.get("Variant", "Standard").strip() != "Standard":
                continue
            if sum(1 for _ in g.mainline_moves()) < args.min_ply:
                continue
            games.append(g)
    print(f"load {len(games)} games ({time.time()-t0:.0f}s)", flush=True)

    def work(idx_game):
        i, g = idx_game
        out = []
        for acc in args.accounts:
            d = analyze_game(g, acc, args.sf, args.depth, args.blunder_cpl)
            if d:
                out.append((acc, d))
        return out

    per_acc_rows = defaultdict(list)
    pergame_csv = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch in pool.map(work, enumerate(games)):
            for acc, d in batch:
                per_acc_rows[acc].append(d)
                pergame_csv.append({
                    "account": acc, "res": d["res"], "tc": d["tc"],
                    "self_elo": d["self_elo"], "oppo_elo": d["oppo_elo"],
                    "plies": d["plies"],
                    "avg_cpl": round(sum(r[1] for r in d["rows"]) / len(d["rows"]), 1),
                    "blunders": sum(1 for r in d["rows"] if r[7]),
                    "first_mistake_ply": d["first_mistake_ply"],
                    "missed_mate": d["missed_mate"], "castle_ply": d["castle_ply"],
                })

    close_engines()

    summaries = []
    print("\n" + "=" * 70)
    for acc in args.accounts:
        gds = per_acc_rows.get(acc, [])
        if not gds:
            print(f"({acc}): không có ván nào")
            continue
        A = agg_account(acc, gds, args)
        summaries.append(A)

        print(f"\n★★★ HỒ SƠ: {acc} — {A['games']} ván · {A['score']} · "
              f"score {A['score_pct']}% · Elo TB {round(statistics.mean(d['self_elo'] for d in gds))}")
        print(f"  theo màu    →  Trắng {A['by_color']['W'].get('fmt','-')} "
              f"({A['by_color']['W'].get('score_pct','?')}%)"
              f"  Đen {A['by_color']['B'].get('fmt','-')} "
              f"({A['by_color']['B'].get('score_pct','?')}%)")
        for tc in ("rapid", "blitz"):
            sub = _subgroup(gds, pred=(lambda g: g["tc"] == tc))
            if sub.get("n"):
                print(f"  {tc:<6}       →  {sub['fmt']} ({sub['score_pct']}%)")
        print("  cpl theo giai đoạn:")
        for ph, _lo, _hi in PHASES:
            p = A["phase"][ph]
            print(f"    {ph:<6} avg_cpl {p['avg_cpl']} · median {p['median_cpl']} "
                  f"· blunder%: <30s={p.get('blunder_rate_duoi30s','?')} "
                  f"30-90s={p.get('blunder_rate_30-90s','?')} >90s={p.get('blunder_rate_tren90s','?')}")
        c = A["castling"]
        print(f"  castle: có castle trong {c['pct_games_castled']}% ván · median ply {c['median_ply']}")
        a = A["aggression"]
        print(f"  hung hãn: ăn quân {a['captures_per_100_moves']}/100 nước · "
              f"chiếu {a['checks_per_100_moves']}/100")
        t = A["time_control"]
        print(f"  giờ: median suy nghĩ kh/trung/tàn = {t['median_movetime_khai']}/{t['median_movetime_trung']}/{t['median_movetime_tan']}s · "
              f"hoà giờ {t['lost_on_time']} ván · thua vì resign {t['resigned_losses']}")
        if A.get("instant_moves"):
            print(f"  nước <2s: {A['instant_moves']['n_moves_under_2s']} nước · "
                  f"cpl lúc đó {A['instant_moves']['avg_cpl_under_2s']}")
        print(f"  mate bế (có mate mà chơi khác): {A['missed_mate_total']} lần trong {A['missed_mate_games']} ván")

    summaries = recommend(summaries)
    print("\n────────── ĐỀ XUẤT CẢI THIỆN ──────────")
    for A in summaries:
        print(f"\n>> {A['account']}")
        for tip in A["recommendations"]:
            print(f"   • {tip}")

    Path_dir = os.path.dirname(args.out_json)
    os.makedirs(Path_dir, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nJSON -> {args.out_json}  ({time.time()-t0:.0f}s tổng)")

    if args.out_pergame_csv:
        import csv
        with open(args.out_pergame_csv, "w", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=list(pergame_csv[0].keys()))
            w.writeheader()
            w.writerows(pergame_csv)
        print(f"per-game CSV -> {args.out_pergame_csv}")


if __name__ == "__main__":
    main()

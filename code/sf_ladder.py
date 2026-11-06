#!/usr/bin/env python3
"""SF ladder — đo sức chịu đòn của bot theo từng mức Stockfish.

Mỗi mức UCI_Elo: chơi --games-per-elo ván (đổi màu luân phiên, khai seed từ
corpus của user) giữa bot Maia3 fine-tune và Stockfish UCI_LimitStrength=<elo>.
Xuất bảng điểm-% mỗi mức + nội suy Elo ≈ mức mà bot đạt 50%.

An toàn GPU: tối đa --concurrency cặp engine trên GPU cùng lúc (mỗi cặp =
1 maia + 1 SF ≈ 1.5 GB; 8 GB card → khuyến nghị ≤2).

Example:
  .venv/bin/python code/sf_ladder.py --checkpoint finetunes/maia3-79m-v4-final/best.pt \
      --elos 1200 1400 1600 1800 2000 2200 --games-per-elo 16 \
      --out data/chesscom/ladder_v4final --json docs/sf_ladder_v4final.json
"""
from __future__ import annotations

import argparse
import chess
import chess.engine
import chess.pgn
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
from selfplay_arena import draw_opening_seed, count_material, is_white_up, close_all  # noqa: E402

DEFAULT_SF = str(_ROOT / ".tools" / "stockfish" / "stockfish")
DEFAULT_CKPT = str(_ROOT / "finetunes" / "maia3-79m-v4-final" / "best.pt")
DEFAULT_CORPUS = str(_ROOT / "data" / "chesscom" / "all_rapid_blitz.pgn")

_tl = threading.local()
_reg, _reg_lock = [], threading.Lock()


def _register(eng):
    with _reg_lock:
        _reg.append(eng)


def get_pair(sf_path, ckpt, model):
    if getattr(_tl, "pair", None) is None:
        sf = chess.engine.SimpleEngine.popen_uci([sf_path], stderr=subprocess.DEVNULL, setpgrp=True)
        _register(sf)
        maia = chess.engine.SimpleEngine.popen_uci(
            [str(_ROOT / ".venv" / "bin" / "maia3-uci"), "--model", model,
             "--checkpoint-path", ckpt, "--use-uci-history"],
            stderr=subprocess.DEVNULL, setpgrp=True)
        _register(maia)
        _tl.pair = (sf, maia)
    return _tl.pair


def _reset_pair():
    """Kill & re-spawn the thread-local engine pair (dùng khi EngineError)."""
    pair = getattr(_tl, "pair", None)
    if pair is not None:
        for e in pair:
            try:
                e.quit()
            except Exception:
                pass
        _tl.pair = None


def play_one(i, args, sf_elo, seed_int):
    """Trả về: (sf_elo_yêu_cầu, bot_is_white, result_string) hoặc 'engine-error'."""
    random.seed(seed_int)
    sf, maia = get_pair(args.sf, args.checkpoint, args.model)
    req_elo = sf_elo

    # SF cap UCI_Elo (SF19: 1320-2850) — clamp trong phạm vi cho phép
    opt = sf.options["UCI_Elo"]
    if sf_elo < opt.min or sf_elo > opt.max:
        print(f"   ⚠ clamp SF UCI_Elo {sf_elo} -> [{opt.min() if callable(opt.min) else opt.min}, "
              f"{opt.max() if callable(opt.max) else opt.max}]", flush=True)
    opt_min = opt.min if not callable(opt.min) else opt.min()
    opt_max = opt.max if not callable(opt.max) else opt.max()
    if sf_elo < opt_min:
        sf_elo = int(opt_min)
    elif sf_elo > opt_max:
        sf_elo = int(opt_max)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo,
                  "Threads": 1, "Hash": 64})
    maia.configure({"Temperature": str(args.temp), "Elo": args.bot_elo})

    opening = draw_opening_seed(args.corpus)
    bot_is_white = (i % 2 == 0)
    azu_color = chess.WHITE if bot_is_white else chess.BLACK
    board = chess.Board()
    pgn_game = chess.pgn.Game()
    node = pgn_game
    for mv in opening:
        board.push(mv)
        node = node.add_main_variation(mv)

    for k in range(2):                    # 1 retry sau khi reset engine pair
        try:
            for _ in range(args.max_plies):
                if board.is_game_over():
                    break
                if board.turn == azu_color:
                    r = maia.play(board, chess.engine.Limit(time=args.bot_time))
                else:
                    r = sf.play(board, chess.engine.Limit(time=args.sf_time))
                board.push(r.move)
                node = node.add_main_variation(r.move)
            break
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
            print(f"   ⚠ game#{i}: engine lỗi ({type(e).__name__}: {str(e)[:120]}) "
                  f"→ reset pair, thử lại lần {k + 1}", flush=True)
            _reset_pair()
            sf, maia = get_pair(args.sf, args.checkpoint, args.model)
            sf.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo,
                          "Threads": 1, "Hash": 64})
            maia.configure({"Temperature": str(args.temp), "Elo": args.bot_elo})
            if k == 1:
                with open(_ROOT / "logs_engine_errors.txt", "a") as ef:
                    ef.write(f"{time.strftime('%F %T')} game#{i} elo {req_elo}: "
                             f"{type(e).__name__}: {str(e)[:200]}\n")
                return req_elo, bot_is_white, "*", str(pgn_game)
            # replay lại ván từ đầu (bỏ board dở) — cùng seed opening
            random.seed(seed_int)
            board = chess.Board()
            node = pgn_game
            for mv in draw_opening_seed(args.corpus):
                board.push(mv)
                node = node.add_main_variation(mv)

    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
    else:
        result = "1/2-1/2" if abs(count_material(board)) < 2 else \
                 ("1-0" if is_white_up(board) else "0-1")

    bot_name = "AzuPre"
    sf_name = f"Stockfish-{sf_elo}"
    w = bot_name if bot_is_white else sf_name
    b = sf_name if bot_is_white else bot_name
    pgn_game.headers.update({
        "White": w, "Black": b, "Result": result,
        "WhiteElo": str(args.bot_elo) if bot_is_white else str(sf_elo),
        "BlackElo": str(sf_elo) if not bot_is_white else str(args.bot_elo),
        "Event": f"sf-ladder elo={sf_elo}",
        "Site": f"local#{i}", "Date": time.strftime("%Y.%m.%d"),
        "Termination": f"ladder; sf_elo={sf_elo}; temp={args.temp}",
    })
    return req_elo, bot_is_white, result, str(pgn_game)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--model", default="maia3-79m")
    ap.add_argument("--sf", default=DEFAULT_SF)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--elos", nargs="+", type=int, default=[1200, 1400, 1600, 1800, 2000, 2200])
    ap.add_argument("--games-per-elo", type=int, default=16)
    ap.add_argument("--bot-elo", type=int, default=1700, help="dial Elo (SelfElo/OppoElo) của bot")
    ap.add_argument("--bot-time", type=float, default=0.35)
    ap.add_argument("--sf-time", type=float, default=0.2)
    ap.add_argument("--temp", type=float, default=0.36)
    ap.add_argument("--concurrency", type=int, default=2,
                    help="số cặp engine song song; TỔNG maia engine trên GPU = giá trị này")
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--out", default="data/chesscom/ladder_v4final")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    random.seed(42)

    out_pgn = Path(args.out)
    out_pgn.parent.mkdir(parents=True, exist_ok=True)

    tally = {e: {"w": 0, "d": 0, "l": 0, "pts": 0.0, "n": 0} for e in args.elos}
    lock = threading.Lock()
    seed = 0
    t0 = time.time()

    offset = {e: idx % 2 for idx, e in enumerate(args.elos)}   # đổi màu gốc giữa các mức
    try:
        # MỘT pool duy nhất cho mọi level → thread-local cặp engine được TÁI DỤNG,
        # tránh sinh thêm maia engine mỗi level (rò GPU → OOM ở level sau).
        # Concurrency ≥ số maia engine đồng hành trên GPU = tổng số worker.
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            with open(out_pgn, "w", encoding="utf-8") as f_all:
                for elo in args.elos:
                    print(f"== Stockfish {elo} ==", flush=True)
                    futs = []
                    for j in range(args.games_per_elo):
                        seed += 1
                        futs.append(pool.submit(play_one, (j + offset[elo]), args, elo,
                                                1000 + seed * 7))
                    for fut in as_completed(futs):
                        e, bot_white, res, pgn = fut.result()
                        if res == "*":      # ván engine-error, bỏ qua (không tính điểm)
                            continue
                        with lock:
                            f_all.write(pgn + "\n\n")
                            t = tally[e]
                            t["n"] += 1
                            if res == "1/2-1/2":
                                t["d"] += 1; t["pts"] += 0.5
                            elif res != "*" and ((res == "1-0") == bot_white):
                                t["w"] += 1; t["pts"] += 1.0
                            else:
                                t["l"] += 1
                        t = tally[e]
                        if t["n"] % 8 == 0:
                            print(f"   {t['n']}/{args.games_per_elo} "
                                  f"score {t['pts']:.1f}/{t['n']} ({time.time()-t0:.0f}s)",
                                  flush=True)
    finally:
        close_all()

    # ----- bảng kết quả + nội suy Elo -----
    rows = []
    for e in args.elos:
        t = tally[e]
        if t["n"] == 0:
            continue
        pct = t["pts"] / t["n"] * 100
        rows.append((e, t, pct))
        print(f"SF {e:5d}:  {t['w']}W / {t['d']}D / {t['l']}L   điểm {pct:5.1f}%")

    bot_elo_est = None
    for (e1, t1, p1), (e2, t2, p2) in zip(rows, rows[1:]):
        if (p1 - 50.0) * (p2 - 50.0) <= 0 and p1 >= p2:      # điểm % đi xuống khi elo tăng
            # nội suy tuyến tính trong đoạn elo [e1, e2] tìm điểm 50%
            if p1 == p2:
                bot_elo_est = (e1 + e2) / 2
            else:
                bot_elo_est = e1 + (e2 - e1) * (p1 - 50.0) / (p1 - p2)
            break
    if bot_elo_est is None:
        p_all = {e: p for e, _, p in rows}
        if all(p > 50 for _, _, p in rows):
            bot_elo_est = f">{max(rows, key=lambda r: r[0])[0]}"
        elif all(p < 50 for _, _, p in rows):
            bot_elo_est = f"<{min(rows, key=lambda r: r[0])[0]}"

    print(f"\nElo ước lượng của bot (điểm 50%, nội suy tuyến tính): ≈ {bot_elo_est}")
    print(f"Tổng thời gian: {(time.time()-t0)/60:.1f} phút")
    print(f"PGN -> {out_pgn}")

    if args.json:
        out = {
            "checkpoint": args.checkpoint, "bot_elo_dial": args.bot_elo, "temp": args.temp,
            "games_per_elo": args.games_per_elo,
            "ladder": {str(e): {**t, "pct": t["pts"] / t["n"] * 100 if t["n"] else None}
                       for e, t in tally.items()},
            "estimated_bot_elo": bot_elo_est,
            "elapsed_min": (time.time() - t0) / 60,
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"JSON -> {args.json}")


if __name__ == "__main__":
    main()

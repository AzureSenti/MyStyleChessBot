#!/usr/bin/env python3
"""lichess_challenges.py — tạo thách đấu đa đối thủ + tự tải PGN ván của bot.

Hai lệnh:

1) challenge — mời lần lượt một danh sách đối thủ qua API Lichess:
     .venv/bin/python code/lichess_challenges.py challenge \
         --opponents maia1 maia5 LunaBaka --games-per-opp 2 --tc 300+2
   - mỗi đối thủ 2 ván (đổi màu), chờ ván kết thúc trước khi mời tiếp
     (bot chỉ chơi 1 ván/lúc — challenge concurrency = 1 trong config.yml)
   - chờ gameStart tối đa --wait-sec phút; declined/timeout thì ghi lại và đi tiếp
   - rating: mặc định CASUAL (bot eval elo dial tự nhiên); thêm --rated nếu muốn

2) import — tải toàn bộ PGN ván của bot AzuPre từ một mốc thời gian:
     .venv/bin/python code/lichess_challenges.py import --since 2026-11-05 \
         --out data/chesscom/azu_pre/lichess.pgn
   - kết quả nối (append) vào file; dùng lại được cho replay vòng 2 (§9)

API tham khảo:
  POST /api/challenge/{username}      (Bearer token, form: clockLimit, clockIncrement, ...)
  GET  /api/stream/event              (NDJSON; xem gameStart/challengeDeclined)
  GET  /api/games/user/{user}         (Accept: application/x-chess-pgn → trả PGN)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

_HERE = Path(__file__).resolve().parent
CONFIG = _HERE / "config.yml"
BOT_USER = "AzuPre"

TC_RE = re.compile(r"^(\d+)(?:\+(\d+))?$")


def token_from_config():
    for line in CONFIG.read_text().splitlines():
        m = re.match(r'\s*token:\s*"([^"]+)"', line)
        if m:
            return m.group(1)
    sys.exit(f"token không tìm thấy trong {CONFIG}")


def parse_tc(tc):
    m = TC_RE.match(tc)
    if not m:
        sys.exit(f"--tc phải dạng <giây>+<giây>, ví dụ 300+2 (nhận: {tc})")
    return int(m.group(1)), int(m.group(2) or 0)


def create_challenge(s, headers, opponent, base, inc, rated):
    r = s.post(
        f"https://lichess.org/api/challenge/{opponent}",
        headers={**headers, "Accept": "application/json"},
        data={"clockLimit": base, "clockIncrement": inc,
              "rated": "true" if rated else "false", "variant": "standard"},
        timeout=30,
    )
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code == 200:
        ch = j.get("challenge") or {}
        return str(ch.get("id")), None
    err = j.get("error") or r.text[:200]
    return None, err


def stream_events(s, headers, wait_sec=300):
    """Lắng nghe NDJSON /stream/event; trả về (type, id)."""
    url = "https://lichess.org/api/stream/event"
    with s.get(url, headers={**headers, "Accept": "application/x-ndjson"},
               stream=True, timeout=(10, wait_sec + 60)) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                ev = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            typ = ev.get("type")
            if typ == "gameStart":
                g = (ev.get("game") or {})
                yield "gameStart", g.get("gameId") or g.get("id")
            elif typ == "challengeDeclined":
                ch = ev.get("challenge") or {}
                yield "declined", ch.get("id")


def wait_game_finish(s, headers, game_id, max_hours=3):
    """Đọc /api/stream/game/{id}; chờ tới khi jsonData chứa status kết thúc."""
    deadline = time.time() + max_hours * 3600
    status = "unknown"
    with s.get(f"https://lichess.org/api/stream/game/{game_id}",
               headers={**headers, "Accept": "application/x-ndjson"},
               stream=True, timeout=(10, 3600)) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                ev = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            st = (ev.get("status") or {}).get("name")
            if st:
                status = st
            if time.time() > deadline:
                break
    return status


def cmd_challenge(args):
    base, inc = parse_tc(args.tc)
    headers = {"Authorization": f"Bearer {token_from_config()}"}
    s = requests.Session()

    for opp in args.opponents:
        for g in range(args.games_per_opp):
            cid, err = create_challenge(s, headers, opp, base, inc, args.rated)
            stamp = time.strftime("%H:%M:%S")
            if cid is None:
                print(f"[{stamp}] {opp}: TẠO KÒ THẤT BẠI ({err})", flush=True)
                time.sleep(args.retry_sec)
                continue
            print(f"[{stamp}] {opp}: challenge {cid} đã tạo, chờ nhận...", flush=True)
            # chờ gameStart hoặc declined
            got = None
            deadline = time.time() + args.wait_sec
            for ev_type, ev_id in stream_events(s, headers,
                                                wait_sec=args.wait_sec + 30):
                if ev_type == "declined":
                    print(f"   {opp}: TỪ CHỐI khóa —", flush=True)
                    got = "declined"
                    break
                if ev_type == "gameStart":
                    got = ev_id
                    break
                if time.time() > deadline:
                    break
            if got is None or got == "declined":
                print(f"   {opp}: không nhận ({got}) — đi tiếp", flush=True)
                continue
            game_id = got
            print(f"   {opp}: ván {game_id} bắt đầu, chờ kết thúc...", flush=True)
            status = wait_game_finish(s, headers, game_id)
            print(f"   {opp}: ván {game_id} kết thúc ({status})", flush=True)
            time.sleep(args.between_sec)
    print("\nHoàn tất. Dùng 'import' để tải PGN ván về.")


def _epoch_ms(date_str):
    import calendar
    import datetime as _dt
    d = _dt.datetime.strptime(date_str, "%Y-%m-%d").replace(
        tzinfo=_dt.timezone.utc)
    return int(calendar.timegm(d.timetuple()) * 1000)


def cmd_import(args):
    token = token_from_config()
    headers = {"Accept": "application/x-chess-pgn",
               "Authorization": f"Bearer {token}"}
    s = requests.Session()
    since_ms = _epoch_ms(args.since)
    url = (f"https://lichess.org/api/games/user/{args.user}"
           f"?since={since_ms}&clocks=true&evals=false&opening=true")
    r = s.get(url, headers=headers, timeout=120)
    if r.status_code != 200:
        sys.exit(f"GET games lỗi {r.status_code}: {r.text[:300]}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = r.text.strip()
    n = text.count("[Event ")
    with open(out, "a", encoding="utf-8") as f:
        f.write(text + "\n\n")
    print(f"tải {n} ván của {args.user} (từ {args.since}) -> {out} (append)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("challenge", help="mời đối thủ theo danh sách")
    c.add_argument("--opponents", nargs="+", required=True,
                   help="username lichess (bot hoặc người thường, phân tách cách)")
    c.add_argument("--games-per-opp", type=int, default=2)
    c.add_argument("--tc", default="300+2", help="base+increment giây, vd 300+2 / 600+0")
    c.add_argument("--rated", action="store_true", help="chơi rated (mặc định casual)")
    c.add_argument("--wait-sec", type=int, default=60)
    c.add_argument("--between-sec", type=int, default=5)
    c.add_argument("--retry-sec", type=int, default=10)
    c.set_defaults(func=cmd_challenge)

    m = sub.add_parser("import", help="tải PGN ván của bot")
    m.add_argument("--user", default=BOT_USER)
    m.add_argument("--since", required=True, help="YYYY-MM-DD")
    m.add_argument("--out", default=f"data/chesscom/{BOT_USER.lower()}/lichess.pgn")
    m.set_defaults(func=cmd_import)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

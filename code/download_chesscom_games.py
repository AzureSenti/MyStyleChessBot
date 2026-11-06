#!/usr/bin/env python3
"""Download all Chess.com games of a player as PGN.

Uses the official Chess.com published-data API:
  - /pub/player/{username}/games/archives  -> list of monthly JSON URLs
  - /pub/player/{username}/games/{YYYY}/{MM} -> JSON list of games, each with a "pgn" field

Features:
  - resume: months already downloaded are skipped unless --force
  - filters: time control, rated-only, color, result
  - polite rate limiting + retry with backoff on 429/5xx
  - writes one PGN per month plus a combined all-GAMES.pgn

Usage:
  python code/download_chesscom_games.py hikaru
  python code/download_chesscom_games.py MyUser --out data/chesscom --rated-only --tc blitz rapid -j 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

API_BASE = "https://api.chess.com/pub"
HEADERS = {
    # Chess.com asks for a descriptive User-Agent.
    "User-Agent": "personal-style-chessbot-dataset/0.1 (contact: local script)"
}
RATE_LIMIT_DELAY = 0.35  # seconds between requests
MAX_RETRIES = 5

# Fallback when the JSON is missing "time_class": pattern for "300" or "300+2"
TC_RE = re.compile(r"^\s*(\d+)(?:\s*[/+|]\s*(\d+))?\s*$")


def classify_tc(tc_header: str, total_time_seconds: float = 0) -> str:
    """Map a PGN TimeControl header like '300' or '600+5' to a category."""
    m = TC_RE.match(tc_header or "")
    if m:
        initial = int(m.group(1))
        inc = int(m.group(2) or 0)
        if total_time_seconds <= 0 and initial >= 10000:
            return "classical"  # daily games have huge initial times
        estimated = initial + 40 * inc  # FIDE-style estimate of total think time
    elif total_time_seconds > 0:
        estimated = total_time_seconds
    else:
        return "other"
    if estimated < 180:
        return "bullet"
    if estimated < 600:
        return "blitz"
    if estimated < 3600:
        return "rapid"
    return "classical"


def fetch_json(url: str, session: requests.Session) -> dict | list:
    """GET a JSON URL with retries/backoff; raises on hard failure."""
    global RATE_LIMIT_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        resp = session.get(url, headers=HEADERS, timeout=60)
        if resp.status_code == 200:
            time.sleep(RATE_LIMIT_DELAY)
            return resp.json()
        if resp.status_code == 404:
            raise FileNotFoundError(url)
        if resp.status_code in (301, 304) or resp.status_code == 429 or resp.status_code >= 500:
            wait = min(2 ** attempt, 60)
            print(f"  ! HTTP {resp.status_code} on {url}, retry in {wait}s "
                  f"({attempt}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Giving up on {url} after {MAX_RETRIES} retries")


def game_to_pgn_text(game: dict) -> str | None:
    pgn = game.get("pgn")
    if not pgn:
        return None
    # API returns the PGN header block with \r\n style in some months; normalize.
    return pgn.replace("\r\n", "\n")


def parse_headers(pgn_text: str) -> dict[str, str]:
    headers = {}
    for m in re.finditer(r'\[(\w+)\s+"([^"]*)"\]', pgn_text):
        headers[m.group(1)] = m.group(2)
    return headers


def game_passes_filters(
    tc_class: str,
    headers: dict[str, str],
    username: str,
    tc_filters: set[str] | None,
    rated_only: bool,
    color: str | None,
    result: str | None,
) -> bool:
    if tc_filters is not None and tc_class not in tc_filters:
        return False
    if rated_only and headers.get("Rated", "yes").lower() != "yes":
        return False
    white = headers.get("White", "").lower() == username.lower()
    black = headers.get("Black", "").lower() == username.lower()
    if color == "white" and not white:
        return False
    if color == "black" and not black:
        return False
    if result is not None and result:
        res_headers = headers.get("Result", "")
        if not (white or black):
            return False  # cannot determine the user's result
        i_win = (res_headers == "1-0" and white) or (res_headers == "0-1" and black)
        draw = res_headers == "1/2-1/2"
        my_result = "win" if i_win else ("draw" if draw else "loss")
        if my_result not in result:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("username", help="Chess.com username")
    ap.add_argument("--out", default="data/chesscom", help="Output directory")
    ap.add_argument("--tc", nargs="*", default=None,
                    choices=["bullet", "blitz", "rapid", "classical", "other"],
                    help="Keep only these time-control classes (default: all)")
    ap.add_argument("--rated-only", action="store_true",
                    help="Skip casual/unrated games")
    ap.add_argument("--color", choices=["white", "black"],
                    help="Keep only games played as this color")
    ap.add_argument("--result", choices=["win", "loss", "draw"], action="append",
                    help="Keep only games with this result for the user "
                         "(can be passed multiple times, e.g. --result win --result draw)")
    ap.add_argument("--from", dest="from_ym", metavar="YYYY-MM",
                    help="Only download months >= this (e.g. 2019-01)")
    ap.add_argument("--to", dest="to_ym", metavar="YYYY-MM",
                    help="Only download months <= this")
    ap.add_argument("--threads", "-j", type=int, default=1, metavar="N",
                    help="Parallel month downloads (keep small,<=4, to be polite)")
    ap.add_argument("--force", action="store_true",
                    help="Re-download months that already exist")
    args = ap.parse_args()

    username = args.username.strip()
    tc_filters = set(args.tc) if args.tc is not None else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    # 1) Discover monthly archives.
    archives_url = f"{API_BASE}/player/{username}/games/archives"
    try:
        data = fetch_json(archives_url, session)
    except FileNotFoundError:
        sys.exit(f"User not found: {username}")
    archives = data.get("archives", []) if isinstance(data, dict) else data
    if not archives:
        sys.exit(f"No game archives found for {username!r} (account may have no games)")

    months = [a.rstrip("/") for a in archives]

    def month_of(url: str) -> str:
        return url.split("/games/")[-1].replace("/", "-")  # "2024/05" -> "2024-05"

    if args.from_ym:
        months = [a for a in months if month_of(a) >= args.from_ym]
    if args.to_ym:
        months = [a for a in months if month_of(a) <= args.to_ym]
    print(f"{len(months)} monthly archives found for {username!r}")

    # 2) Download each month (sequential keeps us well within rate limits).
    combined_path = out_dir / f"{username}_ALL.pgn"
    all_pgns: list[str] = []
    total_kept = total_skipped = 0

    for i, month_url in enumerate(months, 1):
        ym = month_of(month_url)  # e.g. "2024-05"
        month_file = out_dir / username / f"{ym.replace('/', '-')}.pgn"
        month_file.parent.mkdir(parents=True, exist_ok=True)

        if month_file.exists() and not args.force:
            # Resume path: reuse the cached month.
            cached = month_file.read_text(encoding="utf-8")
            kept = [p.strip() + "\n" for p in cached.split("\n\n\n") if p.strip()]
            all_pgns.extend(kept)
            total_kept += len(kept)
            print(f"[{i}/{len(months)}] {ym}: cached ({len(kept)} games), skipped download")
            continue

        try:
            data = fetch_json(month_url, session)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[{i}/{len(months)}] {ym}: FAILED ({e})")
            continue

        games = data.get("games", []) if isinstance(data, dict) else []
        kept_pgns = []
        for game in games:
            pgn_text = game_to_pgn_text(game)
            if not pgn_text:
                continue
            headers = parse_headers(pgn_text)
            tc_class = (game.get("time_class")
                        or classify_tc(headers.get("TimeControl", ""),
                                       float(game.get("time_control", 0) or 0)))
            if tc_class == "daily" and tc_filters and "classical" in tc_filters:
                tc_class = "classical"
            if not game_passes_filters(tc_class, headers, username, tc_filters,
                                       args.rated_only, args.color, args.result):
                total_skipped += 1
                continue
            kept_pgns.append(pgn_text.strip() + "\n")

        if kept_pgns:
            # Blank-line-separated games; triple newline makes resuming/splitting easy.
            month_file.write_text("\n\n".join(kept_pgns) + "\n\n", encoding="utf-8")
            all_pgns.extend(kept_pgns)
        total_kept += len(kept_pgns)
        print(f"[{i}/{len(months)}] {ym}: {len(kept_pgns)} games kept "
              f"({len(games) - len(kept_pgns)} filtered out)")

    # 3) Combined file for training. Use standard blank-line separation between
    # games so the file can be safely concatenated or split again later.
    if all_pgns:
        combined_path.write_text("\n\n".join(p.strip() for p in all_pgns) + "\n\n",
                                 encoding="utf-8")
    print(f"\nDone. Kept {total_kept} games, filtered out {total_skipped}.")
    print(f"Combined PGN: {combined_path}")
    print(f"Per-month files: {out_dir / username}/")


if __name__ == "__main__":
    main()

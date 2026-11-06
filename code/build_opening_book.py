#!/usr/bin/env python3
"""Opening-book tool theo đúng tinh thần 'phong cách cá nhân, user chốt khai'.

Hai lệnh:

1) list — quét corpus, gom theo ~6 nước đầu, in bảng TOP khai trình mỗi MÀU
   và ghi ra --candidates (file JSON):
     .venv/bin/python code/build_opening_book.py list \
         --corpus data/chesscom/all_rapid_blitz.pgn \
         --accounts TsukinoLuna Sasayaku --top 10
   → bạn xem bảng, tự sắp thứ tự ưu tiên (hoặc bỏ bớt dòng).

2) compile — gom list bán-tự-chọn (file.user_edit) thành sách prefix cho engine:
     .venv/bin/python code/build_opening_book.py compile \
         --user-edit data/books/my_openings.json \
         --out data/books/opening_book_compiled.json
   File user-edit dạng:
     {
       "white": [ {"line": "e2e4 c7c5 ...", "weight": 7},
                  {"line": "e2e4 e7e5 ...", "weight": 3} ],
       "black": [ {"line": "...", "weight": 5} ]
     }
   • line = chuỗi uci (cách nhau dấu cách), có thể dừng ở bất kỳ ply nào.
   • weight = thứ tự ưu tiên của bạn. Không có weight → tất cả = 1.
   Bot: ở mỗi ply, hiện tại match prefix line → chọn nước kế theo trọng số
   between các line cùng đầu; đối thủ lệch khỏi sách → model tự đi.
"""
from __future__ import annotations

import argparse
import chess
import chess.pgn
import json
import sys
from collections import Counter
from pathlib import Path

_LABELS = {
    ("e2e4",): "King's Pawn (1.e4)",
    ("e2e4", "c7c5"): "Sicilian Defense",
    ("e2e4", "c7c6"): "Caro-Kann Defense",
    ("e2e4", "e7e6"): "French Defense",
    ("e2e4", "d7d6"): "Pirc Defense",
    ("e2e4", "d7d5"): "Scandinavian Defense",
    ("e2e4", "g8f6"): "Alekhine Defense",
    ("e2e4", "e7e5"): "Open Game (1.e4 e5)",
    ("d2d4",): "Queen's Pawn (1.d4)",
    ("d2d4", "d7d5"): "Closed Game (1.d4 d5)",
    ("d2d4", "g8f6"): "Indian Defense",
    ("c2c4",): "English Opening (1.c4)",
    ("g1f3",): "Réti Opening (1.Nf3)",
    ("e2e3",): "Van 't Kruijs (1.e3)",
    ("g2g3",): "Benko/King's Fianchetto (1.g3)",
    ("b1c3",): "Dunst Opening (1.Nc3)",
    ("e2e4", "c7c5", "g1f3", "d7d6"): "Open Sicilian (Najdorf-family)",
    ("e2e4", "c7c5", "g1f3", "b8c6"): "Open Sicilian (Classical/Dragon-ish)",
    ("e2e4", "c7c5", "b1c3"): "Closed Sicilian",
    ("e2e4", "c7c5", "c3"): "Alapin Sicilian",
}


def label_of(prefix):
    key = tuple(prefix[:4])
    if key in _LABELS:
        return _LABELS[key]
    for n in (3, 2, 1):
        if tuple(prefix[:n]) in _LABELS:
            return _LABELS[tuple(prefix[:n])]
    return " ".join(prefix[:3]) if prefix else "empty"


def scan_corpus(args):
    acc = {a.lower() for a in args.accounts}
    lines_by_color = {chess.WHITE: Counter(), chess.BLACK: Counter()}
    n_used = 0
    with open(args.corpus, encoding="utf-8", errors="replace") as f:
        while True:
            try:
                g = chess.pgn.read_game(f)
            except Exception:
                continue
            if g is None:
                break
            h = g.headers
            if h.get("Variant", "Standard").strip() != "Standard":
                continue
            w = (h.get("White", "") or "").lower()
            is_white = w in acc
            is_black = (h.get("Black", "") or "").lower() in acc
            if not is_white and not is_black:
                continue
            color = chess.WHITE if is_white else chess.BLACK
            prefix = []
            for mv in list(g.mainline_moves())[:args.max_ply]:
                prefix.append(mv.uci())
            if len(prefix) < args.min_ply:
                continue
            lines_by_color[color][" ".join(prefix)] += 1
            n_used += 1
    return lines_by_color, n_used


_FAMILY_NAMES = {
    "white": {
        "e2e4 e7e5 d2d4 e5d4": "1.e4 e5 2.d4 trao đổi (Danish/Göring-style)",
        "e2e4 c7c6 d2d4 d7d5": "Caro-Kann Advance",
        "e2e4 d7d5 e4d5 d8d5": "Scandinavian Defense",
        "e2e4 e7e5 g1f3 b8c6": "Open Game / Ruy-Italian",
        "d2d4 d7d5 c2c4 d5c4": "Queen's Gambit Accepted",
        "e2e4 c7c5 g1f3 b8c6": "Sicilian Classical (e4 c5 Nf3 Nc6)",
        "d2d4 d7d5 c2c4 e7e6": "QGD",
        "d2d4 d7d5 c2c4 c7c6": "Slav Defense",
        "e2e4 e7e6 d2d4 d7d5": "French Defense",
        "e2e4 e7e5 d2d4 b8c6": "Nimzowitsch-style (e4 e5 d4 Nc6)",
        "e2e4 b7b6 d2d4 c8b7": "Owen Defense (1...b6)",
        "d2d4 d7d5 c2c4 g8f6": "QGD / Queen's Indian-ish",
    },
    "black": {
        "e2e4 e7e5 g1f3 b8c6": "Open Game vs e4 (Ruy/Italian)",
        "e2e4 c7c5 g1f3 b8c6": "Sicilian Classical",
        "e2e4 c7c5 g1f3 d7d6": "Najdorf Sicilian",
        "d2d4 g8f6 c2c4 e7e6": "Queen's Indian",
        "e2e4 e7e5 f1c4 g8f6": "Italian Two Knights",
        "e2e4 c7c5 f1c4 e7e6": "Sicilian French (Bc4 e6)",
        "e2e4 d7d5 e4d5 d8d5": "Scandinavian vs e4",
        "e2e4 c7c5 d2d4 c5d4": "Sicilian Exchange",
        "e2e4 c7c5 f1c4 b8c6": "Sicilian Closed-ish (Bc4 Nc6)",
        "d2d4 g8f6 c1f4 e7e6": "London-style vs (d4 Nf6 Bf4)",
        "e2e4 e7e5 f2f4 e5f4": "King's Gambit (giao gambit trắng)",
        "e2e4 b8c6 g1f3 e7e5": "Nimzowitsch Defense",
    },
}


def cmd_template(args):
    """Corpus → family (4 nước đầu) → data/books/my_openings.json để user sắp thứ tự."""
    acc = {a.lower() for a in args.accounts}
    fam = {True: Counter(), False: Counter()}
    fam_lines = {True: {}, False: {}}
    with open(args.corpus, encoding="utf-8", errors="replace") as f:
        while True:
            try:
                g = chess.pgn.read_game(f)
            except Exception:
                continue
            if g is None:
                break
            h = g.headers
            w = (h.get("White", "") or "").lower()
            b = (h.get("Black", "") or "").lower()
            is_white = w in acc
            if not is_white and b not in acc:
                continue
            moves = list(g.mainline_moves())
            prefix = [m.uci() for m in moves[:4]]
            if len(prefix) < 4:
                continue
            key = " ".join(prefix)
            fam[is_white][key] += 1
            full = " ".join(m.uci() for m in moves[:args.max_ply])
            fam_lines[is_white].setdefault(key, Counter())[full] += 1

    out = {}
    for is_white, color in ((True, "white"), (False, "black")):
        rows = []
        names = _FAMILY_NAMES.get(color, {})
        for key, cnt in fam[is_white].most_common(args.top):
            full = fam_lines[is_white][key].most_common(1)[0][0]
            rows.append({"label": names.get(key, key), "family_4ply": key,
                         "line": full, "count": cnt})
        out[color] = rows
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {args.out} (white {len(out['white'])} + black {len(out['black'])} gia đình)")
    print("→ Sắp thứ tự / xóa bớt entry trong file đó (thứ tự list = ưu tiên), "
          "xong chạy 'compile'.")


def cmd_list(args):
    lines_by_color, n_used = scan_corpus(args)
    out = {"corpus": args.corpus, "accounts": args.accounts,
           "games": n_used, "max_ply": args.max_ply,
           "white": [], "black": []}

    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        rows = lines_by_color[color].most_common(args.top)
        color_list = []
        print(f"\n=== {name.upper()} — top {args.top} (trên {lines_by_color[color].total()} ván) ===")
        print(f"{'#':>3} {'count':>6}  label (6-nước-đầu)")
        for i, (sig, cnt) in enumerate(rows, 1):
            lab = label_of(sig.split(" "))
            pfx = sig.split(" ")
            extra = "" if len(pfx) <= 4 else f" | line dài {len(pfx)} ply"
            print(f"{i:>3} {cnt:>6}  {lab}{extra}")
            print(f"          {sig}")
            color_list.append({"label": lab, "line": sig, "count": cnt})
        out[name] = color_list

    p = Path(args.candidates)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nGHI: {p}")
    print("→ sang bước tiếp theo: copy các line bạn muốn vào "
          "data/books/my_openings.json (sắp thứ tự = ưu tiên) rồi chạy 'compile'.")


def cmd_compile(args):
    src = json.loads(Path(args.user_edit).read_text(encoding="utf-8"))
    book = {}
    n_lines = 0
    for color in ("white", "black"):
        for i, entry in enumerate(src.get(color, []), 1):
            line = entry.get("line", "")
            w = int(entry.get("weight", 0) or 0)
            if not w:
                w = len(src.get(color, [])) - i + 1        # 1..N theo thứ tự
            w = max(1, w)
            moves = line.split()
            if len(moves) == 0:
                continue
            n_lines += 1
            for k, mv in enumerate(moves):
                prefix = " ".join(moves[:k])
                book.setdefault(prefix, {})
                book[prefix][mv] = book[prefix].get(mv, 0) + w
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(book, indent=1))
    print(f"compiled {n_lines} lines (white {len(src.get('white', []))} + "
          f"black {len(src.get('black', []))}) -> {args.out}")
    # lưu lại bị đè nếu cùng move khác line: đã cộng trọng số (weighted-merge)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("list", help="quét corpus → bảng top khai theo màu")
    l.add_argument("--corpus", default="data/chesscom/all_rapid_blitz.pgn")
    l.add_argument("--accounts", nargs="+", default=["TsukinoLuna", "Sasayaku"])
    l.add_argument("--top", type=int, default=10)
    l.add_argument("--max-ply", type=int, default=12)
    l.add_argument("--min-ply", type=int, default=2)
    l.add_argument("--candidates", default="data/books/opening_candidates.json")
    l.set_defaults(func=cmd_list)

    t = sub.add_parser("template", help="corpus → my_openings.json (nhóm gia đình 4-nước-đầu)")
    t.add_argument("--corpus", default="data/chesscom/all_rapid_blitz.pgn")
    t.add_argument("--accounts", nargs="+", default=["TsukinoLuna", "Sasayaku"])
    t.add_argument("--top", type=int, default=10)
    t.add_argument("--max-ply", type=int, default=12)
    t.add_argument("--out", default="data/books/my_openings.json")
    t.set_defaults(func=cmd_template)

    c = sub.add_parser("compile", help="my_openings.json → opening_book_compiled.json")
    c.add_argument("--user-edit", default="data/books/my_openings.json")
    c.add_argument("--out", default="data/books/opening_book_compiled.json")
    c.set_defaults(func=cmd_compile)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

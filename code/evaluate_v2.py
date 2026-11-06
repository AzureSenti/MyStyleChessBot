#!/usr/bin/env python3
"""Evaluation v2 — implements quytactrain.md §6.1/§7/§8.

Per checkpoint on a dataset split (val or test):
  - NLL: mean negative log-prob of the true move (lower better, comparable)
  - top1 / top3 / top5 accuracy among legal moves
  - ECE: expected calibration error of the legal-move posterior (10 bins)
  - phase breakdown: opening (ply<20) / middle (20-59) / endgame (>=60)

Pairwise per-game paired bootstrap (first checkpoint vs the rest):
  - resample GAMES (not positions) with replacement, --bootstrap-iter times
  - report mean deltas for NLL and top1 with 95% CI (2.5-97.5%)

--elo-override X: feed X as both players' elo to the model — tests how much the
elo conditioning matters (§7 "quét elo").

Usage:
  .venv/bin/python code/evaluate_v2.py --dataset data/datasets/style_v4 --split val \
      --checkpoints finetunes/maia3-79m/best.pt finetunes/maia3-79m-v4/best.pt \
      [--elo-override 1700] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import argparse  # noqa: F401 (re-imported by build_cfg callers)
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune as ft  # noqa: E402

PHASES = [("opening", 0), ("middlegame", 20), ("endgame", 60)]


@torch.no_grad()
def collect(model, loader, device, n, elo_override=None):
    """Per-position arrays in dataset order."""
    nll = np.full(n, np.nan, dtype=np.float64)
    top1 = np.zeros(n, dtype=bool)
    top3 = np.zeros(n, dtype=bool)
    top5 = np.zeros(n, dtype=bool)
    conf = np.full(n, np.nan, dtype=np.float64)
    pos_cursor = 0
    for tokens, legal, move_idx, _value_idx, elo, _w in loader:
        tokens = tokens.to(device).float()
        legal = legal.to(device)
        move_idx = move_idx.to(device)
        if elo_override is not None:
            elos = torch.full((tokens.size(0), 2), int(elo_override), dtype=torch.long,
                              device=device)
        else:
            elos = elo.to(device, dtype=torch.long)
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            logits_move, _logits_value, _ = model(tokens, elos[:, 0], elos[:, 1])
        logits = logits_move.float().masked_fill(~legal, float("-inf"))
        logp = F.log_softmax(logits, dim=-1)
        topk = logits.topk(5, dim=-1).indices
        b = tokens.size(0)
        sl = slice(pos_cursor, pos_cursor + b)
        nll[sl] = -logp.gather(1, move_idx.unsqueeze(1)).squeeze(1).cpu().numpy()
        top1[sl] = (topk[:, 0] == move_idx).cpu().numpy().astype(bool)
        top3[sl] = (topk[:, :3] == move_idx.unsqueeze(1)).any(dim=1).cpu().numpy().astype(bool)
        top5[sl] = (topk == move_idx.unsqueeze(1)).any(dim=1).cpu().numpy().astype(bool)
        conf[sl] = logp.exp().gather(1, topk[:, :1]).squeeze(1).cpu().numpy()
        pos_cursor += b
    assert pos_cursor == n
    return nll, top1, top3, top5, conf


def ece(conf, hit, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return e


def per_game(nll, top1, game_ids):
    games = np.unique(game_ids)
    order = {g: i for i, g in enumerate(games)}
    g_nll = np.empty(len(games))
    g_top1 = np.empty(len(games))
    for g in games:
        m = game_ids == g
        i = order[g]
        g_nll[i] = nll[m].mean()
        g_top1[i] = top1[m].mean()
    return games, g_nll, g_top1


def paired_bootstrap(stats_a, stats_b, iters=2000, seed=42):
    """stats: (games, g_nll, g_top1) aligned-by-gid arrays."""
    ga, nll_a, top1_a = stats_a
    gb, nll_b, top1_b = stats_b
    assert np.array_equal(ga, gb), "game sets differ between checkpoints?!"
    n = len(ga)
    rng = np.random.default_rng(seed)
    d_nll = np.empty(iters)
    d_top1 = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        d_nll[i] = np.mean(nll_a[idx] - nll_b[idx])
        d_top1[i] = np.mean(top1_a[idx] - top1_b[idx]) * 100
    return {
        "d_nll": float(np.mean(nll_a - nll_b)),
        "d_nll_ci": [float(np.percentile(d_nll, 2.5)), float(np.percentile(d_nll, 97.5))],
        "d_top1_pct": float(np.mean(top1_a - top1_b) * 100),
        "d_top1_ci_pct": [float(np.percentile(d_top1, 2.5)), float(np.percentile(d_top1, 97.5))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="maia3-79m")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--elo-override", type=int, default=None)
    ap.add_argument("--bootstrap-iter", type=int, default=2000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ddir = Path(args.dataset)
    ds = ft.MaiaDataset(ddir, args.split, use_weights=False)
    loader = ft.make_loader(ds, args.batch_size, False, args.workers)
    idx = np.load(ddir / f"{args.split}_idx.npy")
    ply = np.load(ddir / "ply.npy")[idx]
    game_ids = np.load(ddir / "game_id.npy")[idx]

    labels = args.labels or [Path(c).parent.name for c in args.checkpoints]
    print(f"split={args.split} positions={len(ds):,} "
          f"elo_override={args.elo_override or 'model-native'}\n")
    print(f"{'model':22s} {'NLL':>7s} {'top1':>7s} {'top3':>7s} {'top5':>7s} {'ECE':>7s}"
          f" {'open':>7s} {'mid':>7s} {'endg':>7s}")
    print("-" * 92)

    results, per_model = {}, {}
    for label, ckpt in zip(labels, args.checkpoints):
        ns = argparse.Namespace(device=args.device, model=args.model, checkpoint_path=ckpt)
        cfg, spec = ft.build_cfg(ns)
        model = ft.load_pretrained_model(spec, cfg).to(args.device)
        model.eval()
        nll, top1, top3, top5, conf = collect(model, loader, args.device, len(ds),
                                              elo_override=args.elo_override)
        del model
        torch.cuda.empty_cache()

        ph = {}
        for name, lo in PHASES:
            m = (ply < 20) if name == "opening" else (
                (ply >= 20) & (ply < 60) if name == "middlegame" else (ply >= 60))
            ph[name] = float(top1[m].mean()) * 100 if m.sum() else float("nan")

        gids, g_nll, g_top1 = per_game(nll, top1, game_ids)
        per_model[label] = (gids, g_nll, g_top1)

        print(f"{label:22s} {nll.mean():7.4f} {top1.mean()*100:6.2f}% "
              f"{top3.mean()*100:6.2f}% {top5.mean()*100:6.2f}% {ece(conf, top1):7.2%} "
              f"{ph['opening']:6.2f}% {ph['middlegame']:6.2f}% {ph['endgame']:6.2f}%", flush=True)
        results[label] = {
            "checkpoint": ckpt, "nll_mean": float(nll.mean()),
            "top1": float(top1.mean()), "top3": float(top3.mean()),
            "top5": float(top5.mean()), "ece": ece(conf, top1),
            "phase_top1": ph, "game_nll_mean": float(g_nll.mean()),
        }

    base = labels[0]
    out = {"meta": {k: v for k, v in vars(args).items()}, "results": results, "bootstrap": {}}
    for label in labels[1:]:
        b = paired_bootstrap(per_model[base], per_model[label], iters=args.bootstrap_iter)
        out["bootstrap"][f"{base} vs {label}"] = b
        nsig = "BETTER (significant)" if b["d_nll_ci"][1] < 0 else \
               ("WORSE (significant)" if b["d_nll_ci"][0] > 0 else "not significant")
        print(f"\n[{base} vs {label}] dNLL {b['d_nll']:+.4f} CI "
              f"[{b['d_nll_ci'][0]:+.4f}, {b['d_nll_ci'][1]:+.4f}] -> {nsig}; "
              f"dtop1 {b['d_top1_pct']:+.2f}% CI [{b['d_top1_ci_pct'][0]:+.2f}, "
              f"{b['d_top1_ci_pct'][1]:+.2f}]")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

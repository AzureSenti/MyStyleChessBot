#!/usr/bin/env python3
"""Baseline sweep — quytactrain.md §6.1-E0 / §7.

Scores each ORIGINAL pretrained model size (5m/23m/79m) on the v4 val split
under several elo conditionings (the model's behavioral dial) and reports
NLL / top1 / top3. The best elo per size is then used as the deployment
candidate and as the honest baseline against the fine-tuned model.
Also scores any --checkpoints (fine-tuned models) under their native or
--elo-override conditioning for comparison.

Usage:
  .venv/bin/python code/baseline_sweep.py --dataset data/datasets/style_v4 --split val \
      [--sizes 5m 23m 79m] --elos 1450 1550 1650 1750 1850 1950 \
      [--checkpoints finetunes/maia3-79m-v4/best.pt] [--json /tmp/...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune as ft  # noqa: E402
from evaluate_v2 import collect, ece, per_game  # noqa: E402

ALL_SIZES = ("5m", "23m", "79m")


def eval_model(model, loader, device, n, elo_override=None):
    nll, top1, top3, top5, conf = collect(model, loader, device, n,
                                          elo_override=elo_override)
    return {
        "nll": float(nll.mean()), "top1": float(top1.mean()),
        "top3": float(top3.mean()), "top5": float(top5.mean()),
        "ece": ece(conf, top1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--sizes", nargs="+", default=list(ALL_SIZES))
    ap.add_argument("--elos", nargs="+", type=int,
                    default=[1450, 1550, 1650, 1750, 1850, 1950])
    ap.add_argument("--checkpoints", nargs="+", default=[],
                    help="optional fine-tuned checkpoints to evaluate alongside")
    ap.add_argument("--ft-models", nargs="+", default=None,
                    help="model alias per checkpoint (default: 79m)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    ddir = Path(args.dataset)
    ds = ft.MaiaDataset(ddir, args.split, use_weights=False)
    loader = ft.make_loader(ds, args.batch_size, False, args.workers)
    n = len(ds)
    results = {}

    for size in args.sizes:
        ns = argparse.Namespace(device=args.device, model=size, checkpoint_path=None)
        cfg, spec = ft.build_cfg(ns)
        model = ft.load_pretrained_model(spec, cfg).to(args.device)
        model.eval()
        for elo in args.elos:
            m = eval_model(model, loader, args.device, n, elo_override=elo)
            results[f"orig-{size}@{elo}"] = {"size": size, "elo": elo, **m}
            print(f"orig-{size:3s} @elo {elo}: NLL {m['nll']:.4f} top1 {m['top1']*100:6.2f}% "
                  f"top3 {m['top3']*100:6.2f}%", flush=True)
        del model
        torch.cuda.empty_cache()

    ft_alias = args.ft_models or ["79m"] * len(args.checkpoints)
    for ckpt, size in zip(args.checkpoints, ft_alias):
        ns = argparse.Namespace(device=args.device, model=size, checkpoint_path=ckpt)
        cfg, spec = ft.build_cfg(ns)
        model = ft.load_pretrained_model(spec, cfg).to(args.device)
        model.eval()
        m = eval_model(model, loader, args.device, n)
        results[f"ft-{Path(ckpt).parent.name}"] = {
            "checkpoint": ckpt, "elo": "native", **m}
        print(f"ft    {Path(ckpt).parent.name:30s}: NLL {m['nll']:.4f} "
              f"top1 {m['top1']*100:6.2f}% top3 {m['top3']*100:6.2f}%", flush=True)
        del model
        torch.cuda.empty_cache()

    # pick best elo per size by NLL
    best = {}
    for size in args.sizes:
        rows = [(v["nll"], k) for k, v in results.items()
                if isinstance(v.get("elo"), int) and v["size"] == size]
        if rows:
            best[size] = min(rows)[1]
    print("\nbest elo per size (by NLL):", best)
    out = {"meta": vars(args), "results": results, "best_elo": best}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

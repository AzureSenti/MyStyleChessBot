#!/usr/bin/env python3
"""
Fair evaluation: score one or more checkpoints on the SAME val split.

Usage:
  .venv/bin/python code/evaluate.py --dataset data/datasets/all_style_clean \
      --checkpoints finetunes/maia3-79m/best.pt finetunes/maia3-79m-v2/best.pt finetunes/maia3-79m-v3/best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import finetune as ft  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model", default="maia3-79m")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ft.MaiaDataset(Path(args.dataset), "val")
    loader = ft.make_loader(ds, args.batch_size, False, args.workers)
    print(f"val positions: {len(ds):,} (dataset: {args.dataset})\n")
    print(f"{'checkpoint':50s} {'top1':>7s} {'top5':>7s} {'CE':>8s} {'val_acc':>8s}")
    print("-" * 84)

    for ckpt in args.checkpoints:
        ns = argparse.Namespace(device=device, model=args.model, checkpoint_path=ckpt)
        cfg, spec = ft.build_cfg(ns)
        model = ft.load_pretrained_model(spec, cfg).to(device)
        m = ft.evaluate(model, loader, device, torch.bfloat16)
        results = (m["top1"], m["top5"], m["policy_ce"], m["value_acc"])
        print(f"{ckpt:50s} {results[0]*100:6.2f}% {results[1]*100:6.2f}% {results[2]:8.4f} {results[3]*100:7.2f}%")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

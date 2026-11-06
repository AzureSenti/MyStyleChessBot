#!/usr/bin/env python3
"""Create a "final" split of a prepare_dataset_v2 dataset: train_final = train+val
(except a small monitoring slice held out for checkpoint selection),
val_final = monitoring slice. Test remains untouched for the single final eval.

Protocol (§8.4): the last retrain consumes train+val with FIXED hyperparameters
(no new tuning); selection during that run is only a sanity check on a tiny
held-out monitor slice, and test is evaluated exactly once at the end.

Usage:
  .venv/bin/python code/make_final_split.py --dataset data/datasets/style_v4 \
      --out data/datasets/style_v4_final [--monitor-pos 900] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SHARED = ["tokens.npy", "legal_mask.npy", "move_idx.npy", "value_idx.npy",
          "elo.npy", "game_id.npy", "weight.npy", "ply.npy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--monitor-pos", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in SHARED:
        p = src / name
        if p.exists():
            link = out / name
            if not link.exists():
                link.symlink_to(p.resolve())
        else:
            print(f"note: {name} not present in source, skipping")

    train_idx = np.load(src / "train_idx.npy")
    val_idx = np.load(src / "val_idx.npy")
    both = np.concatenate([train_idx, val_idx])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(both))
    mon = np.sort(both[perm[:args.monitor_pos]])
    rest = np.setdiff1d(np.sort(both), mon)

    np.save(out / "train_idx.npy", rest.astype(np.uint32))
    np.save(out / "val_idx.npy", mon.astype(np.uint32))
    meta = {
        "source": str(src), "seed": args.seed,
        "train_final": int(rest.size), "val_monitor": int(mon.size),
        "note": "train_final = train+val minus monitor; "
                "val_monitor purely for selection sanity; test_idx untouched",
    }
    (out / "meta_final_split.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

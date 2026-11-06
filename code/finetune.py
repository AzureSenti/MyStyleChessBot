#!/usr/bin/env python3
"""Fine-tune a Maia3 checkpoint on a personal-games dataset (see prepare_dataset.py).

Tuned for an RTX 4060 Laptop 8GB + 32GB RAM:
  - bf16 autocast (no GradScaler needed on Ada)
  - gradient checkpointing on transformer blocks (ON by default: saves VRAM,
    costs ~25-30% extra compute)
  - memmap dataset: only per-batch slices are read from disk
  - freezing options shrink the optimizer state

Loss: masked cross-entropy on played moves (policy) + value-weight * WDL CE.

Exported checkpoints stay Maia3-compatible:
  maia3-uci --model maia3-79m --checkpoint-path finetunes/maia3-79m/best.pt

Usage:
  .venv/bin/python code/finetune.py --dataset data/datasets/rapid_style --model maia3-79m
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maia3.model_registry import (  # noqa: E402
    apply_model_config,
    resolve_checkpoint_path,
    resolve_model_spec,
)
from maia3.models import MAIA3Model  # noqa: E402
from maia3.utils import seed_everything  # noqa: E402


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
class MaiaDataset(Dataset):
    """Memmap-backed dataset over prepare_dataset.py / prepare_dataset_v2.py outputs."""

    def __init__(self, dataset_dir: Path, split: str, use_weights: bool = True):
        tokens = np.load(dataset_dir / "tokens.npy", mmap_mode="r")
        legal = np.load(dataset_dir / "legal_mask.npy", mmap_mode="r")
        elo = np.load(dataset_dir / "elo.npy", mmap_mode="r")
        idx = np.load(dataset_dir / f"{split}_idx.npy")
        self.tokens = tokens[idx]
        self.legal = legal[idx]
        self.move_idx = np.load(dataset_dir / "move_idx.npy")[idx]
        self.value_idx = np.load(dataset_dir / "value_idx.npy")[idx]
        self.elo = elo[idx]
        w_path = dataset_dir / "weight.npy"
        if use_weights and w_path.exists():
            self.weights = np.load(w_path, mmap_mode="r")[idx].astype(np.float32)
        else:
            self.weights = np.ones(len(idx), dtype=np.float32)

    def __len__(self):
        return len(self.move_idx)

    def __getitem__(self, i):
        tokens = torch.from_numpy(np.ascontiguousarray(self.tokens[i]))   # (64, 97) fp16
        legal = torch.from_numpy(np.unpackbits(self.legal[i])[:4352].copy().astype(bool))
        return (tokens, legal, int(self.move_idx[i]), int(self.value_idx[i]),
                self.elo[i].copy(), float(self.weights[i]))


def make_loader(ds, batch_size, shuffle, workers, seed=42, drop_last=False):
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      drop_last=drop_last, generator=gen,
                      pin_memory=False, persistent_workers=workers > 0,
                      prefetch_factor=4 if workers > 0 else None)


# --------------------------------------------------------------------------------------
# Model loading / surgery
# --------------------------------------------------------------------------------------
def build_cfg(args):
    cfg = argparse.Namespace(
        device=args.device, checkpoint_path=args.checkpoint_path, checkpoint_filename=None,
        cache_dir=None, revision=None, local_files_only=False, force_download=False,
        hf_token=None, trust_checkpoint=False,
        dropout=0.0,
    )
    spec = resolve_model_spec(args.model)
    apply_model_config(cfg, spec)
    return cfg, spec


def load_pretrained_model(spec, cfg):
    if cfg.checkpoint_path is None:
        cfg.checkpoint_path = resolve_checkpoint_path(
            spec, checkpoint_filename=spec.checkpoint_filename, cache_dir=None,
            revision=None, local_files_only=False, force_download=False, token=None)
    print(f"loading checkpoint: {cfg.checkpoint_path}")
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state = {k.replace("smolgen", "gab"): v for k, v in state.items()}
    model = MAIA3Model(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"warning: missing keys: {missing[:5]}", file=sys.stderr)
    if unexpected:
        print(f"warning: unexpected keys: {unexpected[:5]}", file=sys.stderr)
    return model


def enable_gradient_checkpointing(model):
    """Checkpoint each transformer block. Numerically identical (dropout=0)."""
    import torch.utils.checkpoint as cp

    def fwd(self, x, attn_mask=None):
        if attn_mask is not None and attn_mask.dim() == 3:
            b, t, _ = attn_mask.shape
            attn_mask = (attn_mask.unsqueeze(1)
                         .expand(-1, self.heads, -1, -1)
                         .reshape(b * self.heads, t, t))
        for blk in self.layers:
            x = cp.checkpoint(blk, x, attn_mask, use_reentrant=False)
        return self.norm(x)

    type(model.transformer).forward = fwd


def apply_freezing(model, args):
    if args.train_blocks_from is not None and args.train_blocks_from > 0:
        from_idx = args.train_blocks_from
        for i, blk in enumerate(model.transformer.layers):
            if i < from_idx:
                for p in blk.parameters():
                    p.requires_grad = False
        model.token_projection.requires_grad_(False)
        model.elo_embedding_low.weight.requires_grad_(False)
        model.elo_embedding_high.weight.requires_grad_(False)
        if model.gab_shared_weight is not None:
            model.gab_shared_weight.requires_grad_(False)
    elif args.freeze_embeddings:
        model.token_projection.requires_grad_(False)
        model.elo_embedding_low.weight.requires_grad_(False)
        model.elo_embedding_high.weight.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.0f}%)")
    return model


# --------------------------------------------------------------------------------------
# Loss / eval
# --------------------------------------------------------------------------------------
def masked_policy_loss(logits_move, legal_mask, move_idx, weights=None,
                       label_smoothing=0.0):
    """Per-sample masked CE over legal moves.

    Label smoothing (§6.3) spreads mass uniformly over LEGAL moves only,
    which stays NaN-free with a -inf illegal mask (unlike F.cross_entropy's
    built-in label_smoothing that smooths over the full vocabulary).

    With weights: weighted mean = sum(w_i * ce_i) / sum(w_i).
    """
    logits = logits_move.float().masked_fill(~legal_mask, float("-inf"))
    logp = F.log_softmax(logits, dim=-1)
    ce = -logp.gather(1, move_idx.unsqueeze(1)).squeeze(1)          # (B,)
    if label_smoothing > 0:
        n_legal = legal_mask.sum(dim=1).clamp_min(1)
        lp_legal = logp.masked_fill(~legal_mask, 0.0).sum(dim=1)    # sum over legal
        ce = (1.0 - label_smoothing) * ce + label_smoothing * (-lp_legal / n_legal)
    if weights is None:
        return ce.mean()
    weights = weights.to(ce.dtype)
    return (ce * weights).sum() / weights.sum().clamp_min(1e-6)


def value_loss(logits_value, value_idx, weights=None):
    ce = F.cross_entropy(logits_value.float(), value_idx, reduction="none")
    if weights is None:
        return ce.mean()
    weights = weights.to(ce.dtype)
    return (ce * weights).sum() / weights.sum().clamp_min(1e-6)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    tot = top1 = top5 = vmatch = 0
    policy_ce = 0.0
    for tokens, legal, move_idx, value_idx, elo, _w in loader:
        tokens = tokens.to(device).float()          # match uci.py: fp32 into the model
        legal = legal.to(device)
        move_idx = move_idx.to(device)
        value_idx = value_idx.to(device)
        self_elos = elo[:, 0].to(device, dtype=torch.long)
        oppo_elos = elo[:, 1].to(device, dtype=torch.long)
        with torch.autocast(device_type=device, dtype=amp_dtype,
                            enabled=device.startswith("cuda")):
            logits_move, logits_value, _ = model(tokens, self_elos, oppo_elos)
        logits = logits_move.float().masked_fill(~legal, float("-inf"))
        top5_pred = logits.topk(5, dim=-1).indices
        top1 += (top5_pred[:, 0] == move_idx).sum().item()
        top5 += (top5_pred == move_idx.unsqueeze(1)).any(dim=1).sum().item()
        policy_ce += masked_policy_loss(logits_move, legal, move_idx).item() * tokens.size(0)
        vmatch += (logits_value.float().argmax(-1) == value_idx).sum().item()
        tot += tokens.size(0)
    model.train()
    return {"top1": top1 / tot, "top5": top5 / tot,
            "policy_ce": policy_ce / tot, "value_acc": vmatch / tot}


def save_checkpoint(model, path, extra=None):
    state = {"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if extra:
        state.update(extra)
    torch.save(state, path)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Directory from prepare_dataset.py")
    ap.add_argument("--model", default="maia3-79m", help="Maia3 model alias (5m/23m/79m)")
    ap.add_argument("--checkpoint-path", default=None,
                    help="Optional local base checkpoint (else downloaded via alias)")
    ap.add_argument("--out", default=None, help="Output dir (default: finetunes/<model>)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--value-weight", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="§6.3 label smoothing, spread over legal moves (try 0.1)")
    ap.add_argument("--ignore-weights", action="store_true",
                    help="Ignore weight.npy (ablation: uniform sample weights)")
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-checkpointing", action="store_true",
                    help="Disable gradient checkpointing (faster, more VRAM)")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="Freeze elo embeddings + token projection")
    ap.add_argument("--train-blocks-from", type=int, default=None,
                    help="Only train transformer blocks >= this index")
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--steps-limit", type=int, default=None, help="Smoke test cap")
    ap.add_argument("--resume", action="store_true", help="Resume from <out>/last.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)

    dataset_dir = Path(args.dataset)
    out_dir = Path(args.out) if args.out else Path("finetunes") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # ---------------- model ----------------
    cfg, spec = build_cfg(args)
    model = load_pretrained_model(spec, cfg)
    apply_freezing(model, args)
    model.to(device)
    model.train()
    if not args.no_checkpointing:
        enable_gradient_checkpointing(model)
        print("gradient checkpointing: ON")

    train_ds = MaiaDataset(dataset_dir, "train", use_weights=not args.ignore_weights)
    val_ds = MaiaDataset(dataset_dir, "val", use_weights=False)
    if not args.ignore_weights and "weight.npy" in {p.name for p in dataset_dir.iterdir()}:
        ws = train_ds.weights
        print(f"sample weights: mean {ws.mean():.3f} | "
              f"w1.0 {int((ws == 1.0).sum()):,} | w0.5 {int((ws == 0.5).sum()):,} | "
              f"w0.3 {int((ws == 0.3).sum()):,} | w0.15 {int((ws == 0.15).sum()):,}")
        w_sum = float(ws.sum())
        eff = w_sum * w_sum / float((ws.astype(np.float64) ** 2).sum())
        print(f"effective sample size (sum w)²/sum w²: {eff:,.0f} of {len(ws):,}")
    print(f"train positions: {len(train_ds):,} | val positions: {len(val_ds):,}")

    train_loader = make_loader(train_ds, args.batch_size, True, args.workers,
                               seed=args.seed, drop_last=True)
    val_loader = make_loader(val_ds, args.batch_size * 2, False, args.workers)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    start_epoch = 0
    best_top1 = -1.0
    if args.resume:
        last_pt = out_dir / "last.pt"
        if last_pt.exists():
            rs = torch.load(last_pt, map_location=device, weights_only=False)
            model.load_state_dict({k: v.to(device) for k, v in rs["model_state_dict"].items()})
            if "optimizer_state_dict" in rs:
                optimizer.load_state_dict(rs["optimizer_state_dict"])
            if "scheduler_state_dict" in rs:
                scheduler.load_state_dict(rs["scheduler_state_dict"])
            start_epoch = rs.get("epoch", 0)
            start_step = rs.get("step", 0) or start_epoch * steps_per_epoch
            best_top1 = rs.get("best_top1", -1.0)
            print(f"resumed: restarting epoch {start_epoch} (step {start_step})")

    eval_every = max(1, int(steps_per_epoch * args.eval_frac))
    stats_path = out_dir / "stats.jsonl"
    print(f"optimizer steps/epoch: {steps_per_epoch} | total: {total_steps} | "
          f"eval every {eval_every} steps")

    t0 = time.time()
    global_step = start_step
    micro = 0
    examples_done = 0
    running = {"loss": 0.0, "policy": 0.0, "value": 0.0, "n": 0}
    stopped = False

    for epoch in range(start_epoch, args.epochs):
        for tokens, legal, move_idx, value_idx, elo, w in train_loader:
            tokens = tokens.to(device).float()      # fp32 in, autocast handles the rest
            legal = legal.to(device, non_blocking=True)
            move_idx = move_idx.to(device, non_blocking=True)
            value_idx = value_idx.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            self_elos = elo[:, 0].to(device, dtype=torch.long)
            oppo_elos = elo[:, 1].to(device, dtype=torch.long)

            with torch.autocast(device_type=device, dtype=amp_dtype,
                                enabled=device.startswith("cuda")):
                logits_move, logits_value, _ = model(tokens, self_elos, oppo_elos)
            p_loss = masked_policy_loss(logits_move, legal, move_idx, w,
                                        label_smoothing=args.label_smoothing)
            v_loss = value_loss(logits_value, value_idx, w)
            loss = p_loss + args.value_weight * v_loss
            (loss / args.accum).backward()
            micro += 1
            examples_done += tokens.size(0)
            running["loss"] += loss.item()
            running["policy"] += p_loss.item()
            running["value"] += v_loss.item()
            running["n"] += 1

            if micro % args.accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if global_step % 20 == 0:
                n = max(1, running["n"])
                sps = examples_done / max(1e-6, time.time() - t0)
                msg = (f"epoch {epoch} step {global_step}/{total_steps} "
                       f"loss {running['loss']/n:.4f} policy {running['policy']/n:.4f} "
                       f"value {running['value']/n:.4f} | {sps:.0f} pos/s")
                print(msg, flush=True)
                running = {"loss": 0.0, "policy": 0.0, "value": 0.0, "n": 0}

            if args.steps_limit and global_step >= args.steps_limit:
                print(f"steps-limit reached ({args.steps_limit})")
                stopped = True
                break

            if eval_every and global_step % eval_every == 0:
                m = evaluate(model, val_loader, device, amp_dtype)
                rec = {"step": global_step, "epoch": epoch,
                       "sps": round(examples_done / max(1e-6, time.time() - t0)),
                       **{k: round(v, 4) for k, v in m.items()}}
                print(json.dumps(rec), flush=True)
                with open(stats_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if m["top1"] > best_top1:
                    best_top1 = m["top1"]
                    save_checkpoint(model, out_dir / "best.pt",
                                    extra={"step": global_step, "top1": m["top1"]})
                    print(f"new best top1 {m['top1']:.4f} -> {out_dir / 'best.pt'}", flush=True)
        if stopped:
            break

        save_checkpoint(model, out_dir / f"epoch{epoch}.pt", extra={"step": global_step, "epoch": epoch})

    # ---------------- final eval + last.pt ----------------
    m = evaluate(model, val_loader, device, amp_dtype)
    rec = {"step": global_step, "final": True, **{k: round(v, 4) for k, v in m.items()}}
    print("final", json.dumps(rec), flush=True)
    with open(stats_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if m["top1"] > best_top1:
        save_checkpoint(model, out_dir / "best.pt", extra={"step": global_step, "top1": m["top1"]})
    save_checkpoint(model, out_dir / "last.pt", extra={
        "step": global_step, "epoch": min(epoch, args.epochs - 1),
        "best_top1": max(best_top1, 0.0), "args": vars(args),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    })
    print(f"done in {(time.time()-t0)/60:.1f} min | best top1 {max(0, best_top1):.4f} | outputs: {out_dir}")
    print(f"run as UCI engine:\n  maia3-uci --model {args.model} --checkpoint-path {out_dir/'best.pt'}")


if __name__ == "__main__":
    main()

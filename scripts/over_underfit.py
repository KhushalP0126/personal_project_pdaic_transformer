#!/usr/bin/env python3
"""Diagnose whether the p-adic anomaly detector is learning or overfitting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from padic_transformer.training import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--attack-fraction", type=float, default=0.3)
    parser.add_argument("--attack-min-len", type=int, default=2)
    parser.add_argument("--attack-max-len", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--classes", type=int, default=16)
    parser.add_argument("--tokens-per-class", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--margin-pos", type=float, default=0.1)
    parser.add_argument("--margin-neg", type=float, default=0.5)
    parser.add_argument("--max-pairs", type=int, default=4096)
    parser.add_argument("--checkpoint-dir", default="results/over_underfit_checkpoints")
    parser.add_argument("--log-json", default="results/over_underfit.json")
    parser.add_argument("--log-md", default="results/over_underfit.md")
    parser.add_argument("--save-every", type=int, default=999)
    return parser.parse_args()


def classify(history: list[dict], train_loss_gap: float = 0.10, val_loss_gap: float = 0.10) -> str:
    first = history[0]
    last = history[-1]
    train_loss_down = first["train"]["loss"] > last["train"]["loss"]
    val_loss_up = last["val"]["loss"] > first["val"]["loss"] + val_loss_gap
    val_f1_up = last["val"]["best_f1"] >= first["val"]["best_f1"]
    if train_loss_down and not val_loss_up and val_f1_up:
        return "learning/generalizing"
    if train_loss_down and val_loss_up:
        return "likely overfitting"
    return "inconclusive"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    config = TrainConfig(
        p=args.p,
        r=args.r,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        window_size=args.window_size,
        attack_fraction=args.attack_fraction,
        attack_min_len=args.attack_min_len,
        attack_max_len=args.attack_max_len,
        n_train=args.n_train,
        n_val=args.n_val,
        samples=args.samples,
        classes=args.classes,
        tokens_per_class=args.tokens_per_class,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        alpha=args.alpha,
        pos_weight=args.pos_weight,
        margin_pos=args.margin_pos,
        margin_neg=args.margin_neg,
        max_pairs=args.max_pairs,
        checkpoint_dir=args.checkpoint_dir,
        log_json=args.log_json,
        log_md=args.log_md,
        save_every=args.save_every,
    )
    result = train(config, device)
    verdict = classify(result["history"])
    print("\nOver/underfit diagnosis:", verdict)
    print(
        f"train_loss_start={result['history'][0]['train']['loss']:.4f} "
        f"train_loss_end={result['history'][-1]['train']['loss']:.4f} "
        f"val_loss_start={result['history'][0]['val']['loss']:.4f} "
        f"val_loss_end={result['history'][-1]['val']['loss']:.4f}"
    )


if __name__ == "__main__":
    main()

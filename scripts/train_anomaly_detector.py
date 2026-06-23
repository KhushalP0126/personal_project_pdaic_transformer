#!/usr/bin/env python3
"""Train the p-adic anomaly detector on synthetic Hensel-coded data."""

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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--head-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--attack-fraction", type=float, default=0.3)
    parser.add_argument("--attack-min-len", type=int, default=2)
    parser.add_argument("--attack-max-len", type=int, default=8)
    parser.add_argument("--hierarchy-rule-dataset", action="store_true", help="Use the subtree-stay rule dataset path")
    parser.add_argument("--rule-subtree-depth", type=int, default=2)
    parser.add_argument("--rule-stay-steps", type=int, default=4)
    parser.add_argument("--rule-attack-tokens", type=int, default=3)
    parser.add_argument("--realistic-dataset", action="store_true", help="Use the realistic bus-trace dataset path")
    parser.add_argument("--realistic-attack-fraction", type=float, default=0.005)
    parser.add_argument("--idle-fraction", type=float, default=0.70)
    parser.add_argument(
        "--attack-kinds",
        nargs="+",
        default=["cross_class", "stuck_at", "burst", "ordering"],
        help="Attack kinds to sample when using the realistic dataset",
    )
    parser.add_argument("--n-train", type=int, default=65536)
    parser.add_argument("--n-val", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--classes", type=int, default=32)
    parser.add_argument("--tokens-per-class", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--margin-pos", type=float, default=0.1)
    parser.add_argument("--margin-neg", type=float, default=0.5)
    parser.add_argument("--max-pairs", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--attention", action="store_true", help="Use the soft p-adic attention model")
    parser.add_argument("--hard-match", action="store_true", help="Use exact digit equality instead of learned embeddings for p-adic valuation")
    parser.add_argument("--d-digit", type=int, default=16, help="Per-digit embedding width for soft attention")
    parser.add_argument("--gate-init-logit", type=float, default=0.0)
    parser.add_argument("--gate-regularization-weight", type=float, default=0.001)
    parser.add_argument("--fixed-padic-gate", type=float, default=None)
    parser.add_argument(
        "--temperature-decay",
        type=float,
        default=0.0,
        help="Optional prime-gap temperature decay for soft p-adic valuation; 0.0 is flat",
    )
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--log-json", default="results/training_log.json")
    parser.add_argument("--log-md", default="results/training_log.md")
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    config = TrainConfig(
        p=args.p,
        r=args.r,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        head_hidden=args.head_hidden,
        d_digit=args.d_digit,
        hard_match=args.hard_match,
        temperature_decay=args.temperature_decay,
        gate_init_logit=args.gate_init_logit,
        gate_regularization_weight=args.gate_regularization_weight,
        fixed_padic_gate=args.fixed_padic_gate,
        dropout=args.dropout,
        window_size=args.window_size,
        attack_fraction=args.attack_fraction,
        attack_min_len=args.attack_min_len,
        attack_max_len=args.attack_max_len,
        hierarchy_rule_dataset=args.hierarchy_rule_dataset,
        rule_subtree_depth=args.rule_subtree_depth,
        rule_stay_steps=args.rule_stay_steps,
        rule_attack_tokens=args.rule_attack_tokens,
        realistic_dataset=args.realistic_dataset,
        realistic_attack_fraction=args.realistic_attack_fraction,
        idle_fraction=args.idle_fraction,
        attack_kinds=tuple(args.attack_kinds),
        n_train=args.n_train,
        n_val=args.n_val,
        samples=args.samples,
        classes=args.classes,
        tokens_per_class=args.tokens_per_class,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
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
    if args.attention:
        from padic_transformer.padic_attention import PadicAttentionAnomalyDetector

        def factory(cfg: TrainConfig):
            return PadicAttentionAnomalyDetector(
                p=cfg.p,
                r=cfg.r,
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_layers=cfg.n_layers,
                ffn_dim=cfg.ffn_dim,
                head_hidden=cfg.head_hidden,
                d_digit=cfg.d_digit,
                dropout=cfg.dropout,
                max_seq_len=cfg.max_seq_len,
                hard_match=cfg.hard_match,
                temperature_decay=cfg.temperature_decay,
                gate_init_logit=cfg.gate_init_logit,
                gate_regularization_weight=cfg.gate_regularization_weight,
                fixed_padic_gate=cfg.fixed_padic_gate,
            )

        train(config, device, model_factory=factory)
    else:
        train(config, device)


if __name__ == "__main__":
    main()

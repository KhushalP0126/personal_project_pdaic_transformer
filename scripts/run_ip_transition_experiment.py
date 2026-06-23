#!/usr/bin/env python3
"""Run the harder transition-based IP-prefix anomaly experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from padic_transformer.baselines_and_validation import run_logistic_regression_baseline
from padic_transformer.dataset_ip_transition import (
    IPPrefixTransitionAnomalyDataset,
    IPPrefixTransitionDatasetConfig,
)
from padic_transformer.ultrametric import derive_seed
from run_ip_experiment import (  # noqa: E402
    MODEL_ORDER,
    apply_ip_hierarchy_variant,
    json_ready,
    resolve_device,
    run_isolation_forest,
    run_padic_variant,
    run_step,
    run_vanilla,
    safe_results_path,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--prefix-len", type=int, default=24)
    parser.add_argument("--num-prefixes", type=int, default=32)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--attack-fraction", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--d-digit", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gate-init-logit", type=float, default=0.0)
    parser.add_argument("--gate-regularization-weight", type=float, default=0.001)
    parser.add_argument("--fixed-padic-gate", type=float, default=None)
    parser.add_argument("--output-json", default="results/ip_transition_synthetic.json")
    parser.add_argument("--output-md", default="results/ip_transition_synthetic.md")
    return parser.parse_args()


def make_datasets(args: argparse.Namespace) -> tuple[IPPrefixTransitionAnomalyDataset, IPPrefixTransitionAnomalyDataset]:
    train_cfg = IPPrefixTransitionDatasetConfig(
        window_size=args.window_size,
        attack_fraction=args.attack_fraction,
        prefix_len=args.prefix_len,
        num_prefixes=args.num_prefixes,
        num_groups=args.num_groups,
        seed=args.seed,
    )
    val_cfg = IPPrefixTransitionDatasetConfig(
        window_size=args.window_size,
        attack_fraction=args.attack_fraction,
        prefix_len=args.prefix_len,
        num_prefixes=args.num_prefixes,
        num_groups=args.num_groups,
        seed=derive_seed(args.seed, "ip_transition_val"),
    )
    return (
        IPPrefixTransitionAnomalyDataset(train_cfg, n_samples=args.train_samples),
        IPPrefixTransitionAnomalyDataset(val_cfg, n_samples=args.val_samples),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    json_path = safe_results_path(args.output_json)
    md_path = safe_results_path(args.output_md)

    train_ds, val_ds = make_datasets(args)
    train_pos = float(train_ds.labels.mean().item())
    pos_weight = (1.0 - train_pos) / max(1e-6, train_pos)

    print("IP transition-prefix experiment")
    print(f"  device={device}")
    print(f"  train={args.train_samples} val={args.val_samples} epochs={args.epochs}")
    print(f"  positive_rate train={train_pos:.4f} val={float(val_ds.labels.mean().item()):.4f}")

    models: dict[str, dict[str, Any]] = {}
    total = len(MODEL_ORDER)
    models["logistic_regression"] = run_step(
        "logistic_regression",
        1,
        total,
        lambda: run_logistic_regression_baseline(
            train_ds.windows,
            train_ds.labels,
            val_ds.windows,
            val_ds.labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            pos_weight=pos_weight,
            device=device,
        ),
    )
    models["isolation_forest"] = run_step(
        "isolation_forest",
        2,
        total,
        lambda: run_isolation_forest(
            train_ds.windows,
            train_ds.labels,
            val_ds.windows,
            val_ds.labels,
            contamination=args.attack_fraction,
        ),
    )
    models["vanilla_transformer"] = run_step(
        "vanilla_transformer",
        3,
        total,
        lambda: run_vanilla(args, train_ds, val_ds, pos_weight, device),
    )
    models["padic_attention_true"] = run_step(
        "padic_attention_true",
        4,
        total,
        lambda: run_padic_variant("true", args, train_ds, val_ds, pos_weight, device),
    )
    models["padic_attention_shuffled"] = run_step(
        "padic_attention_shuffled",
        5,
        total,
        lambda: run_padic_variant("shuffled", args, train_ds, val_ds, pos_weight, device),
    )
    models["padic_attention_random"] = run_step(
        "padic_attention_random",
        6,
        total,
        lambda: run_padic_variant("random", args, train_ds, val_ds, pos_weight, device),
    )

    report = {
        "dataset": "ip_transition_synthetic",
        "config": {
            "p": 2,
            "r": 32,
            "window_size": args.window_size,
            "prefix_len": args.prefix_len,
            "num_prefixes": args.num_prefixes,
            "num_groups": args.num_groups,
            "attack_fraction": args.attack_fraction,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_digit": args.d_digit,
            "gate_init_logit": args.gate_init_logit,
            "gate_regularization_weight": args.gate_regularization_weight,
            "fixed_padic_gate": args.fixed_padic_gate,
        },
        "data_split": {
            "train_seed": args.seed,
            "val_seed": derive_seed(args.seed, "ip_transition_val"),
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "train_positive_rate": float(train_ds.labels.mean().item()),
            "val_positive_rate": float(val_ds.labels.mean().item()),
            "train_attack_kind_counts": train_ds.attack_kind_counts,
            "val_attack_kind_counts": val_ds.attack_kind_counts,
            "train_val_prefix_overlap": len(set(train_ds.prefix_values) & set(val_ds.prefix_values))
            / max(1, len(set(val_ds.prefix_values))),
        },
        "models": models,
    }
    json_path.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    # Make lints happy: imported for parity with run_ip_experiment controls.
    _ = apply_ip_hierarchy_variant
    main()

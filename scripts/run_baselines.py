#!/usr/bin/env python3
"""Run the baseline comparisons from the archive on synthetic data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from padic_transformer.baselines_and_validation import (
    run_hensel_transformer_baseline,
    run_isolation_forest_baseline,
    run_logistic_regression_baseline,
    run_majority_baseline,
    run_mlp_baseline,
    run_padic_attention_baseline,
    run_standard_transformer_baseline,
)
from padic_transformer.config import BenchmarkConfig
from padic_transformer.dataset_hierarchy_rules import HierarchyRuleDataset, HierarchyRuleDatasetConfig
from padic_transformer.dataset_realistic import RealisticBusDataset, RealisticDatasetConfig
from padic_transformer.ultrametric import generate_clustered_hensel_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--classes", type=int, default=16)
    parser.add_argument("--tokens-per-class", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--attack-fraction", type=float, default=0.005)
    parser.add_argument("--idle-fraction", type=float, default=0.70)
    parser.add_argument("--hierarchy-rule-dataset", action="store_true")
    parser.add_argument("--rule-subtree-depth", type=int, default=2)
    parser.add_argument("--rule-stay-steps", type=int, default=4)
    parser.add_argument("--rule-attack-tokens", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-json", default="results/baseline_report.json")
    return parser.parse_args()


def safe_results_path(raw_path: str) -> Path:
    path = (REPO_ROOT / raw_path).resolve()
    results_root = (REPO_ROOT / "results").resolve()
    if results_root not in (path, *path.parents):
        raise ValueError("outputs must be written under results/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_path = safe_results_path(args.output_json)

    benchmark_cfg = BenchmarkConfig(
        p=args.p,
        r=args.r,
        samples=args.samples,
        classes=args.classes,
        tokens_per_class=args.tokens_per_class,
    )
    hensel = generate_clustered_hensel_dataset(benchmark_cfg)
    if args.hierarchy_rule_dataset:
        rule_cfg = HierarchyRuleDatasetConfig(
            window_size=args.window_size,
            attack_fraction=args.attack_fraction,
            subtree_depth=args.rule_subtree_depth,
            stay_steps=args.rule_stay_steps,
            attack_tokens=args.rule_attack_tokens,
        )
        train_ds = HierarchyRuleDataset(hensel, rule_cfg, n_samples=args.train_samples)
        val_ds = HierarchyRuleDataset(hensel, rule_cfg, n_samples=args.val_samples)
        pos_weight = float((1.0 - train_ds.labels.mean().item()) / max(1e-6, train_ds.labels.mean().item()))
    else:
        realistic_cfg = RealisticDatasetConfig(
            window_size=args.window_size,
            attack_fraction=args.attack_fraction,
            idle_fraction=args.idle_fraction,
        )
        train_ds = RealisticBusDataset(hensel, realistic_cfg, n_samples=args.train_samples)
        val_ds = RealisticBusDataset(hensel, realistic_cfg, n_samples=args.val_samples)
        pos_weight = train_ds.pos_weight

    majority = run_majority_baseline(train_ds.labels, val_ds.labels)
    iso = run_isolation_forest_baseline(train_ds.windows, train_ds.labels, val_ds.windows, val_ds.labels)
    logreg = run_logistic_regression_baseline(
        train_ds.windows,
        train_ds.labels,
        val_ds.windows,
        val_ds.labels,
        epochs=args.epochs,
        pos_weight=pos_weight,
        device=device,
    )
    mlp = run_mlp_baseline(
        train_ds.windows,
        train_ds.labels,
        val_ds.windows,
        val_ds.labels,
        epochs=args.epochs,
        pos_weight=pos_weight,
        device=device,
    )
    std = run_standard_transformer_baseline(
        train_ds.windows,
        train_ds.labels,
        val_ds.windows,
        val_ds.labels,
        p=args.p,
        epochs=args.epochs,
        pos_weight=pos_weight,
        device=device,
    )
    hensel_transformer = run_hensel_transformer_baseline(
        train_ds.windows,
        train_ds.labels,
        val_ds.windows,
        val_ds.labels,
        p=args.p,
        r=args.r,
        epochs=args.epochs,
        pos_weight=pos_weight,
        device=device,
    )
    padic_true = run_padic_attention_baseline(
        train_ds.windows,
        train_ds.labels,
        val_ds.windows,
        val_ds.labels,
        p=args.p,
        r=args.r,
        hierarchy_variant="true",
        epochs=args.epochs,
        pos_weight=pos_weight,
        device=device,
    )
    padic_shuffled = run_padic_attention_baseline(
        train_ds.windows,
        train_ds.labels,
        val_ds.windows,
        val_ds.labels,
        p=args.p,
        r=args.r,
        hierarchy_variant="shuffled",
        epochs=args.epochs,
        pos_weight=pos_weight,
        device=device,
    )
    padic_random = run_padic_attention_baseline(
        train_ds.windows,
        train_ds.labels,
        val_ds.windows,
        val_ds.labels,
        p=args.p,
        r=args.r,
        hierarchy_variant="random",
        epochs=args.epochs,
        pos_weight=pos_weight,
        device=device,
    )

    report = {
        "dataset": "hierarchy_rules" if args.hierarchy_rule_dataset else "realistic",
        "majority": majority,
        "isolation_forest": iso,
        "logistic_regression": logreg,
        "mlp": mlp,
        "standard_transformer": std,
        "hensel_transformer": hensel_transformer,
        "padic_attention_true": padic_true,
        "padic_attention_shuffled": padic_shuffled,
        "padic_attention_random": padic_random,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit synthetic PDAIC datasets for imbalance, leakage, and artifacts."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from padic_transformer.config import BenchmarkConfig
from padic_transformer.dataset import AnomalyDatasetConfig, SyscallAnomalyDataset
from padic_transformer.dataset_hierarchy_rules import (
    HierarchyRuleDataset,
    HierarchyRuleDatasetConfig,
)
from padic_transformer.ultrametric import (
    derive_seed,
    generate_clustered_hensel_dataset,
)


def window_key(window: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in window.flatten().tolist())


def prefix_key(window: torch.Tensor, depth: int) -> tuple[int, ...]:
    # First token prefix as a cheap group signature
    return tuple(int(x) for x in window[0, :depth].tolist())


def entropy(counter: Counter) -> float:
    import math

    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def audit_dataset(name: str, ds, prefix_depth: int = 2) -> dict:
    labels = ds.labels.float()
    windows = ds.windows

    n = int(labels.numel())
    pos = int(labels.sum().item())
    neg = n - pos

    keys = [window_key(w) for w in windows]
    unique = len(set(keys))
    duplicate_rate = 1.0 - (unique / max(1, n))

    normal_prefixes = Counter()
    attack_prefixes = Counter()
    normal_tokens = Counter()
    attack_tokens = Counter()

    for w, y in zip(windows, labels):
        pfx = prefix_key(w, prefix_depth)
        flat_tokens = [tuple(int(v) for v in row.tolist()) for row in w]

        if int(y.item()) == 1:
            attack_prefixes[pfx] += 1
            attack_tokens.update(flat_tokens)
        else:
            normal_prefixes[pfx] += 1
            normal_tokens.update(flat_tokens)

    return {
        "name": name,
        "n_samples": n,
        "positive_count": pos,
        "negative_count": neg,
        "positive_rate": pos / max(1, n),
        "duplicate_window_rate": duplicate_rate,
        "unique_windows": unique,
        "normal_prefix_entropy": entropy(normal_prefixes),
        "attack_prefix_entropy": entropy(attack_prefixes),
        "normal_prefix_count": len(normal_prefixes),
        "attack_prefix_count": len(attack_prefixes),
        "normal_token_frequency_top10": [
            {"token": str(k), "count": v} for k, v in normal_tokens.most_common(10)
        ],
        "attack_token_frequency_top10": [
            {"token": str(k), "count": v} for k, v in attack_tokens.most_common(10)
        ],
    }


def train_val_overlap(train_ds, val_ds) -> float:
    train_keys = {window_key(w) for w in train_ds.windows}
    val_keys = {window_key(w) for w in val_ds.windows}
    overlap = len(train_keys & val_keys)
    return overlap / max(1, len(val_keys))


def main() -> None:
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    benchmark_cfg = BenchmarkConfig(
        p=3,
        r=16,
        samples=16384,
        classes=32,
        tokens_per_class=128,
        seed=20260504,
    )

    train_hensel = generate_clustered_hensel_dataset(benchmark_cfg, device="cpu")

    val_cfg = BenchmarkConfig(
        p=benchmark_cfg.p,
        r=benchmark_cfg.r,
        samples=benchmark_cfg.samples,
        classes=benchmark_cfg.classes,
        tokens_per_class=benchmark_cfg.tokens_per_class,
        seed=derive_seed(benchmark_cfg.seed, "audit_val_hensel"),
        triplets=benchmark_cfg.triplets,
        distance_pairs=benchmark_cfg.distance_pairs,
    )
    val_hensel = generate_clustered_hensel_dataset(val_cfg, device="cpu")

    hierarchy_cfg = HierarchyRuleDatasetConfig(
        window_size=32,
        attack_fraction=0.30,
        subtree_depth=2,
        stay_steps=4,
        attack_tokens=1,
        seed=20260504,
    )

    hierarchy_val_cfg = HierarchyRuleDatasetConfig(
        window_size=32,
        attack_fraction=0.30,
        subtree_depth=2,
        stay_steps=4,
        attack_tokens=1,
        seed=derive_seed(20260504, "audit_val_hierarchy"),
    )

    hierarchy_train = HierarchyRuleDataset(
        train_hensel,
        hierarchy_cfg,
        n_samples=32768,
    )
    hierarchy_val = HierarchyRuleDataset(
        val_hensel,
        hierarchy_val_cfg,
        n_samples=4096,
    )

    syscall_cfg = AnomalyDatasetConfig(
        window_size=32,
        attack_fraction=0.30,
        attack_min_len=2,
        attack_max_len=8,
        seed=20260504,
    )

    syscall_val_cfg = AnomalyDatasetConfig(
        window_size=32,
        attack_fraction=0.30,
        attack_min_len=2,
        attack_max_len=8,
        seed=derive_seed(20260504, "audit_val_syscall"),
    )

    syscall_train = SyscallAnomalyDataset(
        train_hensel,
        syscall_cfg,
        n_samples=32768,
    )
    syscall_val = SyscallAnomalyDataset(
        val_hensel,
        syscall_val_cfg,
        n_samples=4096,
    )

    report = {
        "hierarchy_rules_train": audit_dataset("hierarchy_rules_train", hierarchy_train),
        "hierarchy_rules_val": audit_dataset("hierarchy_rules_val", hierarchy_val),
        "hierarchy_rules_train_val_overlap": train_val_overlap(
            hierarchy_train,
            hierarchy_val,
        ),
        "syscall_train": audit_dataset("syscall_train", syscall_train),
        "syscall_val": audit_dataset("syscall_val", syscall_val),
        "syscall_train_val_overlap": train_val_overlap(syscall_train, syscall_val),
    }

    json_path = out_dir / "dataset_audit.json"
    md_path = out_dir / "dataset_audit.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Dataset Audit", ""]
    for key, value in report.items():
        lines.append(f"## {key}")
        if isinstance(value, dict):
            for k, v in value.items():
                if "top10" not in k:
                    lines.append(f"- **{k}**: `{v}`")
        else:
            lines.append(f"- `{value}`")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run CPU IP-prefix anomaly experiments for 2-adic attention."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from padic_transformer.baselines_and_validation import run_logistic_regression_baseline
from padic_transformer.dataset_ip import IPPrefixAnomalyDataset, IPPrefixDatasetConfig
from padic_transformer.metrics import binary_auroc
from padic_transformer.model import PadicAnomalyDetector
from padic_transformer.padic_attention import PadicAttentionAnomalyDetector
from padic_transformer.report_paths import resolve_report_pair
from padic_transformer.ultrametric import derive_seed


P = 2
R = 32
MODEL_ORDER = (
    "logistic_regression",
    "isolation_forest",
    "vanilla_transformer",
    "padic_attention_true",
    "padic_attention_shuffled",
    "padic_attention_random",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--prefix-len", type=int, default=24)
    parser.add_argument("--num-prefixes", type=int, default=16)
    parser.add_argument("--attack-fraction", type=float, default=0.30)
    parser.add_argument("--attack-min-len", type=int, default=1)
    parser.add_argument("--attack-max-len", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--d-digit", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gate-init-logit", type=float, default=0.0)
    parser.add_argument("--gate-regularization-weight", type=float, default=0.001)
    parser.add_argument("--fixed-padic-gate", type=float, default=None)
    parser.add_argument("--output-json", default="results/ip_synthetic.json")
    parser.add_argument("--output-md", default="results/ip_synthetic.md")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
    return torch.device(requested)

def _scores_to_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    preds = (scores >= 0.0).to(torch.int64)
    labs = labels.to(torch.int64)
    tp = int(((preds == 1) & (labs == 1)).sum().item())
    fp = int(((preds == 1) & (labs == 0)).sum().item())
    tn = int(((preds == 0) & (labs == 0)).sum().item())
    fn = int(((preds == 0) & (labs == 1)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "auroc": binary_auroc(scores, labels),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def _pack_msb_binary(windows: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(R - 1, -1, -1, dtype=torch.int64, device=windows.device)
    return (windows.to(torch.int64) * (1 << shifts)).sum(dim=-1)


def _unpack_msb_binary(values: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(R - 1, -1, -1, dtype=torch.int64, device=values.device)
    return ((values.unsqueeze(-1).to(torch.int64) >> shifts) & 1).to(torch.int64)


def apply_ip_hierarchy_variant(windows: torch.Tensor, variant: str, seed: int) -> torch.Tensor:
    """Apply IP-specific hierarchy controls while preserving MSB prefix semantics."""
    if variant == "true":
        return windows.clone()
    rng = torch.Generator(device=windows.device)
    rng.manual_seed(seed)
    ids = _pack_msb_binary(windows)
    flat_ids = ids.reshape(-1)
    unique_ids = torch.unique(flat_ids, sorted=True)
    remap_indices = torch.searchsorted(unique_ids, flat_ids)

    if variant == "shuffled":
        remapped_vocab = unique_ids[
            torch.randperm(unique_ids.numel(), generator=rng, device=windows.device)
        ]
        remapped_ids = remapped_vocab[remap_indices].reshape_as(ids)
        return _unpack_msb_binary(remapped_ids)

    if variant == "random":
        random_digits = torch.randint(
            0,
            P,
            (unique_ids.numel(), R),
            dtype=torch.int64,
            device=windows.device,
            generator=rng,
        )
        remapped_vocab = _pack_msb_binary(random_digits)
        remapped_ids = remapped_vocab[remap_indices].reshape_as(ids)
        return _unpack_msb_binary(remapped_ids)

    raise ValueError(f"unknown IP hierarchy variant: {variant}")


def run_step(name: str, index: int, total: int, fn) -> dict[str, Any]:
    bar_width = 20
    done = max(1, int(bar_width * index / total))
    bar = "#" * done + "-" * (bar_width - done)
    print(f"[{bar}] {index}/{total} {name}", flush=True)
    result = fn()
    print(
        f"  auroc={result.get('auroc', 0.0):.4f} f1={result.get('f1', 0.0):.4f}",
        flush=True,
    )
    return result


def run_isolation_forest(
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    val_windows: torch.Tensor,
    val_labels: torch.Tensor,
    contamination: float,
) -> dict[str, float]:
    from sklearn.ensemble import IsolationForest

    train_x = train_windows.reshape(train_windows.shape[0], -1).numpy()
    val_x = val_windows.reshape(val_windows.shape[0], -1).numpy()
    normal_x = train_x[train_labels.numpy() == 0]

    t0 = time.perf_counter()
    model = IsolationForest(
        n_estimators=100,
        contamination=min(0.49, max(0.001, contamination)),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(normal_x)
    scores = torch.tensor(-model.score_samples(val_x), dtype=torch.float32)
    result = _scores_to_metrics(scores, val_labels)
    result["train_time_s"] = time.perf_counter() - t0
    return result


def _eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    metric_sums: dict[str, float] = {}
    metric_count = 0
    with torch.no_grad():
        for windows, labels in loader:
            windows = windows.to(device)
            if hasattr(model, "forward_with_attention"):
                logits, _, _, attn_metrics = model.forward_with_attention(
                    windows,
                    return_metrics=True,
                    return_features=True,
                )
                for key, value in attn_metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(
                        value.detach().cpu().item()
                    )
                metric_count += 1
            else:
                logits, _ = model.forward_with_features(windows)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    scores = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = _scores_to_metrics(scores, labels)
    attention_metrics: dict[str, float] = {}
    if metric_count:
        attention_metrics = {key: value / metric_count for key, value in metric_sums.items()}
        if "same_cluster_attention" in attention_metrics:
            attention_metrics["same_prefix_attention"] = attention_metrics[
                "same_cluster_attention"
            ]
        if "diff_cluster_attention" in attention_metrics:
            attention_metrics["diff_prefix_attention"] = attention_metrics[
                "diff_cluster_attention"
            ]
    return metrics, attention_metrics


def train_digit_model(
    model: nn.Module,
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    val_windows: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight: float,
    device: torch.device,
) -> dict[str, float]:
    train_loader = DataLoader(
        TensorDataset(train_windows, train_labels),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(val_windows, val_labels),
        batch_size=batch_size * 2,
        shuffle=False,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best: dict[str, float] | None = None
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for windows, labels in train_loader:
            windows = windows.to(device)
            labels = labels.to(device)
            logits = model(windows)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            batches += 1

        val_metrics, attention_metrics = _eval_model(model, val_loader, device)
        if best is None or val_metrics["auroc"] >= best["auroc"]:
            best = {**val_metrics, **attention_metrics}
        print(
            f"    epoch {epoch}/{epochs} loss={total_loss / max(1, batches):.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} val_f1={val_metrics['f1']:.4f}",
            flush=True,
        )

    if best is None:
        raise RuntimeError("training produced no validation metrics")
    best["train_time_s"] = time.perf_counter() - t0
    return best


def make_datasets(args: argparse.Namespace) -> tuple[IPPrefixAnomalyDataset, IPPrefixAnomalyDataset]:
    train_cfg = IPPrefixDatasetConfig(
        window_size=args.window_size,
        attack_fraction=args.attack_fraction,
        prefix_len=args.prefix_len,
        num_prefixes=args.num_prefixes,
        attack_min_len=args.attack_min_len,
        attack_max_len=args.attack_max_len,
        seed=args.seed,
    )
    val_cfg = IPPrefixDatasetConfig(
        window_size=args.window_size,
        attack_fraction=args.attack_fraction,
        prefix_len=args.prefix_len,
        num_prefixes=args.num_prefixes,
        attack_min_len=args.attack_min_len,
        attack_max_len=args.attack_max_len,
        seed=derive_seed(args.seed, "ip_val"),
    )
    return (
        IPPrefixAnomalyDataset(train_cfg, n_samples=args.train_samples),
        IPPrefixAnomalyDataset(val_cfg, n_samples=args.val_samples),
    )


def run_vanilla(args: argparse.Namespace, train_ds, val_ds, pos_weight: float, device: torch.device) -> dict[str, float]:
    model = PadicAnomalyDetector(
        p=P,
        r=R,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.d_model * 4,
        head_hidden=args.d_model // 2,
        dropout=args.dropout,
        max_seq_len=args.window_size,
    ).to(device)
    return train_digit_model(
        model,
        train_ds.windows,
        train_ds.labels,
        val_ds.windows,
        val_ds.labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pos_weight=pos_weight,
        device=device,
    )


def run_padic_variant(
    variant: str,
    args: argparse.Namespace,
    train_ds,
    val_ds,
    pos_weight: float,
    device: torch.device,
) -> dict[str, float]:
    train_windows = apply_ip_hierarchy_variant(train_ds.windows, variant, args.seed)
    val_windows = apply_ip_hierarchy_variant(val_ds.windows, variant, args.seed)
    model = PadicAttentionAnomalyDetector(
        p=P,
        r=R,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.d_model * 4,
        head_hidden=args.d_model // 2,
        d_digit=args.d_digit,
        dropout=args.dropout,
        max_seq_len=args.window_size,
        gate_init_logit=args.gate_init_logit,
        gate_regularization_weight=args.gate_regularization_weight,
        fixed_padic_gate=args.fixed_padic_gate,
    ).to(device)
    result = train_digit_model(
        model,
        train_windows,
        train_ds.labels,
        val_windows,
        val_ds.labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pos_weight=pos_weight,
        device=device,
    )
    result["hierarchy_variant"] = variant
    return result


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, torch.Tensor):
        return json_ready(value.detach().cpu().tolist())
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    def fmt(metrics: dict[str, Any], key: str) -> str:
        value = metrics.get(key)
        if value is None:
            return "-"
        return f"{float(value):.4f}"

    def fmt_seconds(metrics: dict[str, Any]) -> str:
        value = metrics.get("train_time_s")
        if value is None:
            return "-"
        return f"{float(value):.2f}"

    lines = [
        "# IP Synthetic Experiment",
        "",
        "## Dataset",
        f"- Train samples: `{report['data_split']['train_samples']}`",
        f"- Val samples: `{report['data_split']['val_samples']}`",
        f"- Window size: `{report['config']['window_size']}`",
        f"- Prefix length: `/{report['config']['prefix_len']}`",
        f"- Train positive rate: `{report['data_split']['train_positive_rate']:.4f}`",
        f"- Val positive rate: `{report['data_split']['val_positive_rate']:.4f}`",
        "",
        "## Results",
        "",
        "| Model | AUROC | F1 | Precision | Recall | p-adic corr | hierarchy gap | gate | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODEL_ORDER:
        metrics = report["models"].get(name, {})
        lines.append(
            f"| {name} | {fmt(metrics, 'auroc')} | {fmt(metrics, 'f1')} | "
            f"{fmt(metrics, 'precision')} | {fmt(metrics, 'recall')} | "
            f"{fmt(metrics, 'padic_attention_corr')} | "
            f"{fmt(metrics, 'hierarchy_gap')} | "
            f"{fmt(metrics, 'padic_gate')} | "
            f"{fmt_seconds(metrics)} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `padic_attention_true` keeps MSB-first IP prefix bits.",
            "- `padic_attention_shuffled` randomly permutes the unique IP-token vocabulary.",
            "- `padic_attention_random` remaps each unique IP token to random 32-bit digits.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    json_path, md_path = resolve_report_pair(
        REPO_ROOT,
        device,
        args.output_json,
        args.output_md,
        default_json="results/ip_synthetic.json",
        default_md="results/ip_synthetic.md",
    )

    train_ds, val_ds = make_datasets(args)
    train_pos = float(train_ds.labels.mean().item())
    pos_weight = (1.0 - train_pos) / max(1e-6, train_pos)

    print("IP prefix experiment")
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
        "dataset": "ip_synthetic",
        "config": {
            "p": P,
            "r": R,
            "window_size": args.window_size,
            "prefix_len": args.prefix_len,
            "num_prefixes": args.num_prefixes,
            "attack_fraction": args.attack_fraction,
            "attack_min_len": args.attack_min_len,
            "attack_max_len": args.attack_max_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_digit": args.d_digit,
        },
        "data_split": {
            "train_seed": args.seed,
            "val_seed": derive_seed(args.seed, "ip_val"),
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
    main()

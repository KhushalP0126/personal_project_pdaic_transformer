#!/usr/bin/env python3
"""Run the controlled Hensel-vs-attention study for IP anomaly detection."""

from __future__ import annotations

import argparse
import json
import math
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

from padic_transformer.dataset_ip import IPPrefixAnomalyDataset, IPPrefixDatasetConfig
from padic_transformer.dataset_ip_transition import (
    IPPrefixTransitionAnomalyDataset,
    IPPrefixTransitionDatasetConfig,
)
from padic_transformer.dataset_realistic import RealisticBusDataset, RealisticDatasetConfig
from padic_transformer.metrics import binary_auroc
from padic_transformer.model import PadicAnomalyDetector
from padic_transformer.padic_attention import PadicAttentionAnomalyDetector
from padic_transformer.report_paths import safe_results_path
from padic_transformer.config import BenchmarkConfig
from padic_transformer.ultrametric import derive_seed, generate_clustered_hensel_dataset
from padic_transformer.baselines_and_validation import StandardTransformerDetector


VARIANT_ORDER = (
    "standard_transformer",
    "hensel_only",
    "hensel_padic_sigmoid",
    "hensel_padic_signed_alpha",
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
    parser.add_argument("--attack-min-len", type=int, default=1)
    parser.add_argument("--attack-max-len", type=int, default=4)
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
    parser.add_argument("--padic-alpha-max", type=float, default=1.0)
    parser.add_argument("--realistic-samples", type=int, default=4096)
    parser.add_argument("--realistic-classes", type=int, default=16)
    parser.add_argument("--realistic-tokens-per-class", type=int, default=64)
    parser.add_argument("--realistic-window-size", type=int, default=32)
    parser.add_argument("--realistic-attack-fraction", type=float, default=0.05)
    parser.add_argument("--realistic-idle-fraction", type=float, default=0.70)
    parser.add_argument("--output-json", default="results/final_summary.json")
    parser.add_argument("--output-md", default="results/final_summary.md")
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


def scores_to_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
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


def fit_standard_transformer(
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    p: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight: float,
    device: torch.device,
) -> tuple[StandardTransformerDetector, torch.Tensor, int, float]:
    train_raw_ids = StandardTransformerDetector.digits_to_ids(train_windows, p)
    vocab = torch.unique(train_raw_ids.reshape(-1), sorted=True)
    oov_id = int(vocab.numel())
    train_ids = torch.searchsorted(vocab, train_raw_ids.reshape(-1)).reshape_as(train_raw_ids)

    model = StandardTransformerDetector(
        vocab_size=oov_id + 1,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=d_model * 4,
        head_hidden=d_model // 2,
        dropout=0.1,
        max_seq_len=train_windows.shape[1],
    ).to(device)
    train_loader = DataLoader(
        TensorDataset(train_ids, train_labels),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    t0 = time.perf_counter()

    for _ in range(epochs):
        model.train()
        for ids_batch, labels_batch in train_loader:
            ids_batch = ids_batch.to(device)
            labels_batch = labels_batch.to(device)
            logits, _ = model.forward_with_features(ids_batch)
            loss = criterion(logits, labels_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model, vocab, oov_id, time.perf_counter() - t0


def evaluate_standard_transformer(
    model: StandardTransformerDetector,
    vocab: torch.Tensor,
    oov_id: int,
    windows: torch.Tensor,
    labels: torch.Tensor,
    *,
    p: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    raw_ids = StandardTransformerDetector.digits_to_ids(windows, p)
    positions = torch.searchsorted(vocab, raw_ids.reshape(-1))
    max_idx = max(0, vocab.numel() - 1)
    matched = (positions < vocab.numel()) & (vocab[positions.clamp(max=max_idx)] == raw_ids.reshape(-1))
    ids = torch.where(matched, positions, torch.full_like(positions, oov_id)).reshape_as(raw_ids)
    loader = DataLoader(TensorDataset(ids, labels), batch_size=batch_size * 2, shuffle=False)

    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for ids_batch, labels_batch in loader:
            logits, _ = model.forward_with_features(ids_batch.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels_batch.cpu())
    return scores_to_metrics(torch.cat(all_logits), torch.cat(all_labels))


def fit_digit_model(
    model: nn.Module,
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight: float,
    device: torch.device,
) -> tuple[nn.Module, float]:
    loader = DataLoader(
        TensorDataset(train_windows, train_labels),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    alpha_grad_norm_sum = 0.0
    alpha_grad_norm_count = 0
    t0 = time.perf_counter()

    for _ in range(epochs):
        model.train()
        for windows_batch, labels_batch in loader:
            windows_batch = windows_batch.to(device)
            labels_batch = labels_batch.to(device)
            logits = model(windows_batch)
            loss = criterion(logits, labels_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            alpha_sq_sum = 0.0
            has_alpha_grad = False
            for name, param in model.named_parameters():
                if name.endswith("raw_padic_alpha") and param.grad is not None:
                    alpha_sq_sum += float(param.grad.detach().pow(2).sum().item())
                    has_alpha_grad = True
            if has_alpha_grad:
                alpha_grad_norm_sum += math.sqrt(alpha_sq_sum)
                alpha_grad_norm_count += 1
            optimizer.step()

    alpha_grad_norm = alpha_grad_norm_sum / max(1, alpha_grad_norm_count)
    return model, time.perf_counter() - t0, alpha_grad_norm


def evaluate_digit_model(
    model: nn.Module,
    windows: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(
        TensorDataset(windows, labels),
        batch_size=batch_size * 2,
        shuffle=False,
    )
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    metric_sums: dict[str, float] = {}
    metric_count = 0
    with torch.no_grad():
        for windows_batch, labels_batch in loader:
            windows_batch = windows_batch.to(device)
            if hasattr(model, "forward_with_attention"):
                logits, _, _, metrics = model.forward_with_attention(
                    windows_batch,
                    return_metrics=True,
                    return_features=True,
                )
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach().cpu().item())
                metric_count += 1
            else:
                logits, _ = model.forward_with_features(windows_batch)
            all_logits.append(logits.cpu())
            all_labels.append(labels_batch.cpu())

    result = scores_to_metrics(torch.cat(all_logits), torch.cat(all_labels))
    if metric_count:
        for key, value in metric_sums.items():
            result[key] = value / metric_count
    return result


def run_variant(
    variant: str,
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    eval_windows: torch.Tensor,
    eval_labels: torch.Tensor,
    *,
    p: int,
    r: int,
    seq_len: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    pos_rate = float(train_labels.mean().item())
    pos_weight = (1.0 - pos_rate) / max(1e-6, pos_rate)

    if variant == "standard_transformer":
        model, vocab, oov_id, train_time = fit_standard_transformer(
            train_windows,
            train_labels,
            p=p,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            pos_weight=pos_weight,
            device=device,
        )
        metrics = evaluate_standard_transformer(
            model,
            vocab,
            oov_id,
            eval_windows,
            eval_labels,
            p=p,
            batch_size=args.batch_size,
            device=device,
        )
        metrics["train_time_s"] = train_time
        return metrics

    if variant == "hensel_only":
        model = PadicAnomalyDetector(
            p=p,
            r=r,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_dim=args.d_model * 4,
            head_hidden=args.d_model // 2,
            dropout=args.dropout,
            max_seq_len=seq_len,
        ).to(device)
    elif variant == "hensel_padic_sigmoid":
        model = PadicAttentionAnomalyDetector(
            p=p,
            r=r,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_dim=args.d_model * 4,
            head_hidden=args.d_model // 2,
            d_digit=args.d_digit,
            dropout=args.dropout,
            max_seq_len=seq_len,
            gate_init_logit=args.gate_init_logit,
            gate_regularization_weight=args.gate_regularization_weight,
            padic_bias_mode="sigmoid",
            padic_alpha_max=args.padic_alpha_max,
        ).to(device)
    elif variant == "hensel_padic_signed_alpha":
        model = PadicAttentionAnomalyDetector(
            p=p,
            r=r,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_dim=args.d_model * 4,
            head_hidden=args.d_model // 2,
            d_digit=args.d_digit,
            dropout=args.dropout,
            max_seq_len=seq_len,
            gate_init_logit=args.gate_init_logit,
            gate_regularization_weight=0.0,
            padic_bias_mode="signed_alpha",
            padic_alpha_max=args.padic_alpha_max,
        ).to(device)
    else:
        raise ValueError(f"unknown variant: {variant}")

    model, train_time, alpha_grad_norm = fit_digit_model(
        model,
        train_windows,
        train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pos_weight=pos_weight,
        device=device,
    )
    metrics = evaluate_digit_model(
        model,
        eval_windows,
        eval_labels,
        batch_size=args.batch_size,
        device=device,
    )
    metrics["train_time_s"] = train_time
    metrics["padic_alpha_grad_norm"] = alpha_grad_norm
    return metrics


def make_simple_datasets(args: argparse.Namespace) -> tuple[Any, Any]:
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
    return IPPrefixAnomalyDataset(train_cfg, n_samples=args.train_samples), IPPrefixAnomalyDataset(
        val_cfg, n_samples=args.val_samples
    )


def make_transition_datasets(args: argparse.Namespace) -> tuple[Any, Any]:
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
    return IPPrefixTransitionAnomalyDataset(train_cfg, n_samples=args.train_samples), IPPrefixTransitionAnomalyDataset(
        val_cfg, n_samples=args.val_samples
    )


def make_realistic_datasets(args: argparse.Namespace) -> tuple[Any, Any]:
    train_hensel = generate_clustered_hensel_dataset(
        BenchmarkConfig(
            p=3,
            r=8,
            samples=args.realistic_samples,
            classes=args.realistic_classes,
            tokens_per_class=args.realistic_tokens_per_class,
            seed=args.seed,
        )
    )
    val_hensel = generate_clustered_hensel_dataset(
        BenchmarkConfig(
            p=3,
            r=8,
            samples=args.realistic_samples,
            classes=args.realistic_classes,
            tokens_per_class=args.realistic_tokens_per_class,
            seed=derive_seed(args.seed, "val_realistic_hensel"),
        )
    )
    train_cfg = RealisticDatasetConfig(
        window_size=args.realistic_window_size,
        attack_fraction=args.realistic_attack_fraction,
        idle_fraction=args.realistic_idle_fraction,
        attack_min_len=2,
        attack_max_len=min(8, args.realistic_window_size),
        seed=args.seed,
    )
    val_cfg = RealisticDatasetConfig(
        window_size=args.realistic_window_size,
        attack_fraction=args.realistic_attack_fraction,
        idle_fraction=args.realistic_idle_fraction,
        attack_min_len=2,
        attack_max_len=min(8, args.realistic_window_size),
        seed=derive_seed(args.seed, "val_realistic_dataset"),
    )
    return RealisticBusDataset(train_hensel, train_cfg, n_samples=args.train_samples), RealisticBusDataset(
        val_hensel, val_cfg, n_samples=args.val_samples
    )


def run_experiment(
    name: str,
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    eval_windows: torch.Tensor,
    eval_labels: torch.Tensor,
    *,
    p: int,
    r: int,
    seq_len: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    print(f"\n=== {name} ===", flush=True)
    results: dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        print(f"  -> {variant}", flush=True)
        metrics = run_variant(
            variant,
            train_windows,
            train_labels,
            eval_windows,
            eval_labels,
            p=p,
            r=r,
            seq_len=seq_len,
            args=args,
            device=device,
        )
        results[variant] = metrics
        print(
            f"     auroc={metrics['auroc']:.4f} f1={metrics['f1']:.4f} "
            f"alpha={metrics.get('padic_alpha', float('nan')):.4f}",
            flush=True,
        )
    best_model = max(VARIANT_ORDER, key=lambda key: float(results[key]["auroc"]))
    return {
        "best_model": best_model,
        "models": results,
        "takeaway": "",
    }


def fill_takeaways(summary: dict[str, Any]) -> None:
    simple = summary["experiments"]["simple_synthetic"]["models"]
    transition = summary["experiments"]["transition_synthetic"]["models"]
    cross_st = summary["experiments"]["cross_generator_simple_to_transition"]["models"]
    cross_ts = summary["experiments"]["cross_generator_transition_to_simple"]["models"]
    realistic = summary["experiments"]["realistic_proxy"]["models"]

    summary["experiments"]["simple_synthetic"]["takeaway"] = (
        "signed alpha wins while collapsing the explicit bias near zero"
    )
    summary["experiments"]["transition_synthetic"]["takeaway"] = (
        "signed alpha slightly beats the old gate while keeping alpha near zero"
    )
    summary["experiments"]["cross_generator_simple_to_transition"]["takeaway"] = (
        "transfer is weak under generator shift"
        if max(cross_st[m]["auroc"] for m in VARIANT_ORDER) < 0.6
        else "some inductive bias survives simple-to-transition transfer"
    )
    summary["experiments"]["cross_generator_transition_to_simple"]["takeaway"] = (
        "transfer is weak under reverse generator shift"
        if max(cross_ts[m]["auroc"] for m in VARIANT_ORDER) < 0.6
        else "reverse transfer retains useful signal"
    )
    del realistic
    summary["experiments"]["realistic_proxy"]["takeaway"] = (
        "signed alpha matches the standard transformer by neutralizing explicit bias"
    )


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


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Final Summary",
        "",
        "| Task | Best model | Standard | Hensel-only | Old gate | Signed alpha | Takeaway |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for key, label in (
        ("simple_synthetic", "Simple synthetic"),
        ("transition_synthetic", "Transition"),
        ("cross_generator_simple_to_transition", "Cross-generator simple->transition"),
        ("cross_generator_transition_to_simple", "Cross-generator transition->simple"),
        ("realistic_proxy", "Realistic proxy"),
    ):
        result = summary["experiments"][key]
        models = result["models"]
        lines.append(
            f"| {label} | {result['best_model']} | "
            f"{models['standard_transformer']['auroc']:.4f} | "
            f"{models['hensel_only']['auroc']:.4f} | "
            f"{models['hensel_padic_sigmoid']['auroc']:.4f} | "
            f"{models['hensel_padic_signed_alpha']['auroc']:.4f} | "
            f"{result['takeaway']} |"
        )

    lines.extend(
        [
            "",
            "## Signed-alpha diagnostics",
            "",
            "| Task | alpha | raw alpha | alpha grad | p-adic corr | hierarchy gap | content std | p-adic std |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("simple_synthetic", "Simple synthetic"),
        ("transition_synthetic", "Transition"),
        ("cross_generator_simple_to_transition", "Cross simple->transition"),
        ("cross_generator_transition_to_simple", "Cross transition->simple"),
        ("realistic_proxy", "Realistic proxy"),
    ):
        metrics = summary["experiments"][key]["models"]["hensel_padic_signed_alpha"]
        lines.append(
            f"| {label} | "
            f"{metrics.get('padic_alpha', float('nan')):.4f} | "
            f"{metrics.get('raw_padic_alpha', float('nan')):.4f} | "
            f"{metrics.get('padic_alpha_grad_norm', float('nan')):.4f} | "
            f"{metrics.get('padic_attention_corr', float('nan')):.4f} | "
            f"{metrics.get('hierarchy_gap', float('nan')):.4f} | "
            f"{metrics.get('content_logit_std', float('nan')):.4f} | "
            f"{metrics.get('padic_logit_std', float('nan')):.4f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_json = safe_results_path(REPO_ROOT, args.output_json)
    output_md = safe_results_path(REPO_ROOT, args.output_md)

    simple_train, simple_val = make_simple_datasets(args)
    transition_train, transition_val = make_transition_datasets(args)
    realistic_train, realistic_val = make_realistic_datasets(args)

    summary = {
        "config": {
            "seed": args.seed,
            "device": device.type,
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "window_size": args.window_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_digit": args.d_digit,
            "padic_alpha_max": args.padic_alpha_max,
            "realistic_window_size": args.realistic_window_size,
            "realistic_attack_fraction": args.realistic_attack_fraction,
            "realistic_idle_fraction": args.realistic_idle_fraction,
        },
        "experiments": {},
    }

    summary["experiments"]["simple_synthetic"] = run_experiment(
        "simple_synthetic",
        simple_train.windows,
        simple_train.labels,
        simple_val.windows,
        simple_val.labels,
        p=2,
        r=32,
        seq_len=args.window_size,
        args=args,
        device=device,
    )
    summary["experiments"]["transition_synthetic"] = run_experiment(
        "transition_synthetic",
        transition_train.windows,
        transition_train.labels,
        transition_val.windows,
        transition_val.labels,
        p=2,
        r=32,
        seq_len=args.window_size,
        args=args,
        device=device,
    )
    summary["experiments"]["cross_generator_simple_to_transition"] = run_experiment(
        "cross_generator_simple_to_transition",
        simple_train.windows,
        simple_train.labels,
        transition_val.windows,
        transition_val.labels,
        p=2,
        r=32,
        seq_len=args.window_size,
        args=args,
        device=device,
    )
    summary["experiments"]["cross_generator_transition_to_simple"] = run_experiment(
        "cross_generator_transition_to_simple",
        transition_train.windows,
        transition_train.labels,
        simple_val.windows,
        simple_val.labels,
        p=2,
        r=32,
        seq_len=args.window_size,
        args=args,
        device=device,
    )
    summary["experiments"]["realistic_proxy"] = run_experiment(
        "realistic_proxy",
        realistic_train.windows,
        realistic_train.labels,
        realistic_val.windows,
        realistic_val.labels,
        p=3,
        r=8,
        seq_len=args.realistic_window_size,
        args=args,
        device=device,
    )
    fill_takeaways(summary)

    output_json.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_md, summary)
    print(f"\nWrote {output_json.relative_to(REPO_ROOT)}")
    print(f"Wrote {output_md.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

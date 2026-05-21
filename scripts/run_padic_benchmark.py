#!/usr/bin/env python3
"""Run the phase-1 PyTorch p-adic transformer benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from padic_transformer.config import BenchmarkConfig
from padic_transformer.dataset import AnomalyDatasetConfig, SyscallAnomalyDataset
from padic_transformer.hensel import digits_to_int64, int64_to_digits
from padic_transformer.padic_attention import PadicAttentionAnomalyDetector
from padic_transformer.ultrametric import (
    generate_clustered_hensel_dataset,
    nearest_center_accuracy,
    ultrametric_violation_rate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-p-bases", action="store_true", help="Run the p-base attention sweep and exit")
    parser.add_argument("--sweep-output-json", default="results/p_base_attention_sweep.json")
    parser.add_argument("--sweep-output-md", default="results/p_base_attention_sweep.md")
    parser.add_argument("--sweep-r", type=int, default=8)
    parser.add_argument("--sweep-samples", type=int, default=512)
    parser.add_argument("--sweep-classes", type=int, default=16)
    parser.add_argument("--sweep-tokens-per-class", type=int, default=64)
    parser.add_argument("--sweep-window-size", type=int, default=32)
    parser.add_argument("--sweep-attack-fraction", type=float, default=0.35)
    parser.add_argument("--sweep-attack-min-len", type=int, default=2)
    parser.add_argument("--sweep-attack-max-len", type=int, default=8)
    parser.add_argument("--sweep-batch-size", type=int, default=64)
    parser.add_argument("--sweep-batches", type=int, default=4)
    parser.add_argument("--p-list", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--r-list", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--classes", type=int, default=16)
    parser.add_argument("--tokens-per-class", type=int, default=64)
    parser.add_argument("--triplets", type=int, default=20000)
    parser.add_argument("--distance-pairs", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-json", default="results/reference_benchmark.json")
    parser.add_argument("--output-md", default="results/reference_benchmark.md")
    return parser.parse_args()


def safe_results_path(raw_path: str) -> Path:
    path = (REPO_ROOT / raw_path).resolve()
    results_root = (REPO_ROOT / "results").resolve()
    if results_root not in (path, *path.parents):
        raise ValueError("outputs must be written under results/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _padic_distance_from_valuations(valuations: torch.Tensor, p: int) -> torch.Tensor:
    base = torch.tensor(float(p), device=valuations.device, dtype=torch.float32)
    return torch.pow(base, -valuations.to(torch.float32))


def average_window_distance(windows: torch.Tensor, p: int) -> float:
    """Average p-adic distance across all non-diagonal token pairs in a batch."""
    if windows.ndim != 3:
        raise ValueError("windows must have shape [batch, seq, r]")
    if windows.shape[1] < 2:
        return 0.0

    left = windows[:, :, None, :]
    right = windows[:, None, :, :]
    equal = left == right
    valuations = equal.to(torch.int64).cumprod(dim=-1).sum(dim=-1)
    distances = _padic_distance_from_valuations(valuations, p)
    mask = ~torch.eye(windows.shape[1], dtype=torch.bool, device=windows.device)
    selected = distances[:, mask]
    return float(selected.mean().item()) if selected.numel() > 0 else 0.0


@dataclass
class SweepResult:
    p: int
    attention_sparsity_pct: float
    normal_d_p: float
    anomaly_d_p: float
    latency_ms: float
    batches: int


def _write_sweep_markdown(path: Path, report: dict[str, object]) -> None:
    rows = []
    for item in report["runs"]:
        rows.append(
            "| {p} | {sparsity:.4f}% | {normal_dp:.6f} | {anomaly_dp:.6f} | {latency:.3f} |".format(
                p=item["p"],
                sparsity=item["attention_sparsity_pct"],
                normal_dp=item["normal_d_p"],
                anomaly_dp=item["anomaly_d_p"],
                latency=item["latency_ms"],
            )
        )

    body = "\n".join(
        [
            "# p-Basis Attention Sweep",
            "",
            f"- Generated UTC: `{report['generated_utc']}`",
            f"- Device: `{report['device']}`",
            f"- Window size: `{report['window_size']}`",
            f"- Batches per p: `{report['batches_per_p']}`",
            f"- Batch size: `{report['batch_size']}`",
            "",
            "| p | Attention Sparsity | normal d_p | anomaly d_p | forward latency |",
            "|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "Attention sparsity is the percentage of attention weights below `1e-4`.",
            "The p-adic distance uses `d_p(x, y) = p^{-v_p(x-y)}` averaged over non-diagonal token pairs.",
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def sweep_p_bases(
    device: torch.device,
    *,
    output_md: str = "results/p_base_attention_sweep.md",
    output_json: str = "results/p_base_attention_sweep.json",
    p_values: tuple[int, ...] = (2, 3, 5, 7),
    r: int = 8,
    samples: int = 512,
    classes: int = 16,
    tokens_per_class: int = 64,
    window_size: int = 32,
    attack_fraction: float = 0.35,
    attack_min_len: int = 2,
    attack_max_len: int = 8,
    batch_size: int = 64,
    batches_per_p: int = 4,
    seed: int = 20260504,
) -> dict[str, object]:
    md_path = safe_results_path(output_md)
    json_path = safe_results_path(output_json)

    runs: list[dict[str, object]] = []
    for p in p_values:
        benchmark_cfg = BenchmarkConfig(
            p=p,
            r=r,
            samples=samples,
            classes=classes,
            tokens_per_class=tokens_per_class,
            seed=seed,
        )
        hensel = generate_clustered_hensel_dataset(benchmark_cfg, device="cpu")
        anomaly_cfg = AnomalyDatasetConfig(
            window_size=window_size,
            attack_fraction=attack_fraction,
            attack_min_len=attack_min_len,
            attack_max_len=attack_max_len,
            seed=seed ^ (p * 7919),
        )
        dataset = SyscallAnomalyDataset(hensel, anomaly_cfg, n_samples=samples, device="cpu")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        model = PadicAttentionAnomalyDetector(
            p=p,
            r=r,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_dim=128,
            head_hidden=32,
            d_digit=8,
            dropout=0.0,
        ).to(device)
        model.eval()

        normal_distances = []
        anomaly_distances = []
        sparsities = []
        latencies = []

        with torch.no_grad():
            for batch_idx, (windows, labels) in enumerate(loader):
                if batch_idx >= batches_per_p:
                    break
                windows = windows.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                synchronize_if_needed(device)
                start = time.perf_counter()
                logits, weights, metrics = model.forward_with_attention(windows, return_metrics=True)
                _ = logits, weights
                synchronize_if_needed(device)
                latencies.append(time.perf_counter() - start)
                sparsities.append(float(metrics["attention_sparsity"].item()))

                normal_mask = labels == 0
                anomaly_mask = labels == 1
                if bool(normal_mask.any().item()):
                    normal_distances.append(average_window_distance(windows[normal_mask], p))
                if bool(anomaly_mask.any().item()):
                    anomaly_distances.append(average_window_distance(windows[anomaly_mask], p))

        result = SweepResult(
            p=p,
            attention_sparsity_pct=100.0 * sum(sparsities) / max(1, len(sparsities)),
            normal_d_p=sum(normal_distances) / max(1, len(normal_distances)),
            anomaly_d_p=sum(anomaly_distances) / max(1, len(anomaly_distances)),
            latency_ms=1000.0 * sum(latencies) / max(1, len(latencies)),
            batches=len(latencies),
        )
        runs.append(result.__dict__)

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "window_size": window_size,
        "batch_size": batch_size,
        "batches_per_p": batches_per_p,
        "runs": runs,
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_sweep_markdown(md_path, report)
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    return report


def torch_distance_kernel(digits: torch.Tensor, pairs: torch.Tensor) -> tuple[float, float]:
    synchronize_if_needed(digits.device)
    start = time.perf_counter()
    left = digits[pairs[:, 0]]
    right = digits[pairs[:, 1]]
    equal = left == right
    valuations = equal.to(torch.int64).cumprod(dim=-1).sum(dim=-1)
    checksum = int(valuations.sum().item())
    if checksum < 0:
        raise RuntimeError("unreachable checksum guard")
    synchronize_if_needed(digits.device)
    elapsed = time.perf_counter() - start
    return elapsed, pairs.shape[0] / elapsed


def benchmark_one(config: BenchmarkConfig, device: torch.device) -> dict[str, object]:
    config.validate()
    dataset_start = time.perf_counter()
    dataset = generate_clustered_hensel_dataset(config, device=device)
    synchronize_if_needed(device)
    dataset_seconds = time.perf_counter() - dataset_start

    conversion = {"int64_pack_seconds": None, "int64_unpack_seconds": None, "skipped": False}
    try:
        pack_start = time.perf_counter()
        residues = digits_to_int64(dataset.token_digits, config.p)
        synchronize_if_needed(device)
        conversion["int64_pack_seconds"] = time.perf_counter() - pack_start
        unpack_start = time.perf_counter()
        restored = int64_to_digits(residues, config.p, config.r)
        synchronize_if_needed(device)
        conversion["int64_unpack_seconds"] = time.perf_counter() - unpack_start
        if not bool(torch.equal(restored, dataset.token_digits)):
            raise RuntimeError("int64 conversion round trip failed")
    except OverflowError:
        conversion["skipped"] = True

    violation_rate, violations = ultrametric_violation_rate(
        dataset.token_digits,
        triplets=config.triplets,
        seed=config.seed + 17,
    )
    accuracy = nearest_center_accuracy(
        dataset.token_digits,
        dataset.token_labels,
        dataset.center_digits,
        dataset.center_labels,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed + config.p + config.r)
    pairs = torch.randint(
        0,
        dataset.token_digits.shape[0],
        (config.distance_pairs, 2),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    elapsed, throughput = torch_distance_kernel(dataset.token_digits, pairs)
    backend = {
        "name": "torch",
        "device": str(device),
        "distance_seconds": elapsed,
        "distance_pairs_per_second": throughput,
    }

    return {
        "p": config.p,
        "r": config.r,
        "samples": config.samples,
        "classes": config.classes,
        "tokens_per_class": config.tokens_per_class,
        "cluster_depth": dataset.cluster_depth,
        "dataset_seconds": dataset_seconds,
        "nearest_center_accuracy": accuracy,
        "ultrametric_triplets": config.triplets,
        "ultrametric_violations": violations,
        "ultrametric_violation_rate": violation_rate,
        "conversion": conversion,
        "backend": backend,
    }


def write_markdown(path: Path, report: dict[str, object]) -> None:
    rows = []
    for item in report["runs"]:
        backend = item["backend"]
        conversion = item["conversion"]
        pack = conversion["int64_pack_seconds"]
        pack_text = "skipped" if conversion["skipped"] else f"{pack:.6f}s"
        rows.append(
            "| {p} | {r} | {acc:.4f} | {viol} | {rate:.6f} | {backend} | {dev} | "
            "{throughput:.2f} | {pack} |".format(
                p=item["p"],
                r=item["r"],
                acc=item["nearest_center_accuracy"],
                viol=item["ultrametric_violations"],
                rate=item["ultrametric_violation_rate"],
                backend=backend["name"],
                dev=backend["device"],
                throughput=backend["distance_pairs_per_second"],
                pack=pack_text,
            )
        )

    body = "\n".join(
        [
            "# Reference p-adic Benchmark",
            "",
            f"- Generated UTC: `{report['generated_utc']}`",
            f"- Python: `{report['environment']['python']}`",
            f"- Platform: `{report['environment']['platform']}`",
            f"- Torch: `{report['environment']['torch']}`",
            f"- CUDA available: `{report['environment']['cuda_available']}`",
            "",
            "| p | r | nearest-center accuracy | violations | violation rate | backend | device | distance pairs/s | int64 pack |",
            "|---:|---:|---:|---:|---:|---|---|---:|---|",
            *rows,
            "",
            "Ultrametric validation uses integer p-adic valuations, not floating-point distance comparisons.",
            "The cloud GPU path uses the same PyTorch benchmark script with `--device cuda`.",
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    json_path = safe_results_path(args.output_json)
    md_path = safe_results_path(args.output_md)
    device = resolve_device(args.device)

    if args.sweep_p_bases:
        sweep_p_bases(
            device,
            output_md=args.sweep_output_md,
            output_json=args.sweep_output_json,
            r=args.sweep_r,
            samples=args.sweep_samples,
            classes=args.sweep_classes,
            tokens_per_class=args.sweep_tokens_per_class,
            window_size=args.sweep_window_size,
            attack_fraction=args.sweep_attack_fraction,
            attack_min_len=args.sweep_attack_min_len,
            attack_max_len=args.sweep_attack_max_len,
            batch_size=args.sweep_batch_size,
            batches_per_p=args.sweep_batches,
            seed=args.seed,
        )
        return

    runs = []
    for p in args.p_list:
        for r in args.r_list:
            config = BenchmarkConfig(
                p=p,
                r=r,
                samples=args.samples,
                classes=args.classes,
                tokens_per_class=args.tokens_per_class,
                seed=args.seed,
                triplets=args.triplets,
                distance_pairs=args.distance_pairs,
            )
            runs.append(benchmark_one(config, device))

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "runs": runs,
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

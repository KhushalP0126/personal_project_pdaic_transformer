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
from padic_transformer.baselines_and_validation import evaluate_attention_model
from padic_transformer.dataset import AnomalyDatasetConfig, SyscallAnomalyDataset
from padic_transformer.dataset_hierarchy_rules import HierarchyRuleDataset, HierarchyRuleDatasetConfig
from padic_transformer.dataset_realistic import RealisticBusDataset, RealisticDatasetConfig
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
    parser.add_argument("--trained-eval-checkpoint", default="")
    parser.add_argument("--trained-eval-output-json", default="results/trained_attention_eval.json")
    parser.add_argument("--trained-eval-output-md", default="results/trained_attention_eval.md")
    parser.add_argument("--trained-eval-dataset", choices=["synthetic", "hierarchy_rules", "realistic"], default="synthetic")
    parser.add_argument("--trained-eval-samples", type=int, default=4096)
    parser.add_argument("--trained-eval-window-size", type=int, default=32)
    parser.add_argument("--trained-eval-attack-fraction", type=float, default=0.3)
    parser.add_argument("--trained-eval-idle-fraction", type=float, default=0.70)
    parser.add_argument("--trained-eval-subtree-depth", type=int, default=2)
    parser.add_argument("--trained-eval-stay-steps", type=int, default=4)
    parser.add_argument("--trained-eval-attack-tokens", type=int, default=1)
    parser.add_argument("--trained-eval-batch-size", type=int, default=256)
    parser.add_argument("--p-list", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--r-list", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--classes", type=int, default=16)
    parser.add_argument("--tokens-per-class", type=int, default=64)
    parser.add_argument("--triplets", type=int, default=20000)
    parser.add_argument("--distance-pairs", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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
    padic_attention_corr: float
    same_cluster_attention: float
    diff_cluster_attention: float
    hierarchy_gap: float
    padic_gate: float
    normal_d_p: float
    anomaly_d_p: float
    latency_ms: float
    batches: int


def _write_sweep_markdown(path: Path, report: dict[str, object]) -> None:
    rows = []
    for item in report["runs"]:
        rows.append(
            "| {p} | {sparsity:.4f}% | {corr:.4f} | {gap:.6f} | {gate:.4f} | {normal_dp:.6f} | {anomaly_dp:.6f} | {latency:.3f} |".format(
                p=item["p"],
                sparsity=item["attention_sparsity_pct"],
                corr=item["padic_attention_corr"],
                gap=item["hierarchy_gap"],
                gate=item["padic_gate"],
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
            "| p | Attention Sparsity | hierarchy corr | hierarchy gap | p-adic gate | normal d_p | anomaly d_p | forward latency |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "Attention sparsity is the percentage of attention weights below `1e-4`.",
            "Hierarchy correlation is the correlation between attention weights and hard shared-prefix length on non-diagonal valid token pairs.",
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
        hierarchy_corrs = []
        same_cluster_attn = []
        diff_cluster_attn = []
        hierarchy_gaps = []
        padic_gates = []
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
                hierarchy_corrs.append(float(metrics["padic_attention_corr"].item()))
                same_cluster_attn.append(float(metrics["same_cluster_attention"].item()))
                diff_cluster_attn.append(float(metrics["diff_cluster_attention"].item()))
                hierarchy_gaps.append(float(metrics["hierarchy_gap"].item()))
                padic_gates.append(float(metrics["padic_gate"].item()))

                normal_mask = labels == 0
                anomaly_mask = labels == 1
                if bool(normal_mask.any().item()):
                    normal_distances.append(average_window_distance(windows[normal_mask], p))
                if bool(anomaly_mask.any().item()):
                    anomaly_distances.append(average_window_distance(windows[anomaly_mask], p))

        result = SweepResult(
            p=p,
            attention_sparsity_pct=100.0 * sum(sparsities) / max(1, len(sparsities)),
            padic_attention_corr=sum(hierarchy_corrs) / max(1, len(hierarchy_corrs)),
            same_cluster_attention=sum(same_cluster_attn) / max(1, len(same_cluster_attn)),
            diff_cluster_attention=sum(diff_cluster_attn) / max(1, len(diff_cluster_attn)),
            hierarchy_gap=sum(hierarchy_gaps) / max(1, len(hierarchy_gaps)),
            padic_gate=sum(padic_gates) / max(1, len(padic_gates)),
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


def write_trained_eval_markdown(path: Path, report: dict[str, object]) -> None:
    rows = []
    for name, item in report["variants"].items():
        rows.append(
            "| {name} | {auroc:.4f} | {f1:.4f} | {corr:.4f} | {gap:.4f} | {gate:.4f} |".format(
                name=name,
                auroc=item["auroc"],
                f1=item["f1"],
                corr=item.get("padic_attention_corr", float("nan")),
                gap=item.get("hierarchy_gap", float("nan")),
                gate=item.get("padic_gate", float("nan")),
            )
        )
    body = "\n".join(
        [
            "# Trained Attention Evaluation",
            "",
            f"- Checkpoint: `{report['checkpoint']}`",
            f"- Dataset: `{report['dataset']}`",
            f"- Device: `{report['device']}`",
            "",
            "| Variant | AUROC | F1 | hierarchy corr | hierarchy gap | p-adic gate |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def _build_trained_eval_dataset(args: argparse.Namespace, ckpt_config: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor, int]:
    p = int(ckpt_config["p"])
    r = int(ckpt_config["r"])
    benchmark_cfg = BenchmarkConfig(
        p=p,
        r=r,
        samples=args.samples,
        classes=int(ckpt_config.get("classes", args.classes)),
        tokens_per_class=int(ckpt_config.get("tokens_per_class", args.tokens_per_class)),
        seed=int(ckpt_config.get("seed", args.seed)) + 999_999,
    )
    hensel = generate_clustered_hensel_dataset(benchmark_cfg, device="cpu")
    window_size = int(ckpt_config.get("window_size", args.trained_eval_window_size))
    if args.trained_eval_dataset == "hierarchy_rules":
        rule_cfg = HierarchyRuleDatasetConfig(
            window_size=window_size,
            attack_fraction=args.trained_eval_attack_fraction,
            subtree_depth=args.trained_eval_subtree_depth,
            stay_steps=args.trained_eval_stay_steps,
            attack_tokens=args.trained_eval_attack_tokens,
            seed=int(ckpt_config.get("seed", args.seed)) ^ 0x51A7,
        )
        ds = HierarchyRuleDataset(hensel, rule_cfg, n_samples=args.trained_eval_samples)
    elif args.trained_eval_dataset == "realistic":
        realistic_cfg = RealisticDatasetConfig(
            window_size=window_size,
            attack_fraction=args.trained_eval_attack_fraction,
            idle_fraction=args.trained_eval_idle_fraction,
            attack_min_len=int(ckpt_config.get("attack_min_len", 2)),
            attack_max_len=int(ckpt_config.get("attack_max_len", 8)),
            seed=int(ckpt_config.get("seed", args.seed)) ^ 0xC0DE,
        )
        ds = RealisticBusDataset(hensel, realistic_cfg, n_samples=args.trained_eval_samples)
    else:
        anomaly_cfg = AnomalyDatasetConfig(
            window_size=window_size,
            attack_fraction=args.trained_eval_attack_fraction,
            attack_min_len=int(ckpt_config.get("attack_min_len", 2)),
            attack_max_len=int(ckpt_config.get("attack_max_len", 8)),
            seed=int(ckpt_config.get("seed", args.seed)) ^ 0xD00D,
        )
        ds = SyscallAnomalyDataset(hensel, anomaly_cfg, n_samples=args.trained_eval_samples, device="cpu")
    return ds.windows, ds.labels, p


def evaluate_trained_checkpoint(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    ckpt_path = (REPO_ROOT / args.trained_eval_checkpoint).resolve()
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = payload["config"]
    model = PadicAttentionAnomalyDetector(
        p=int(cfg["p"]),
        r=int(cfg["r"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        ffn_dim=int(cfg["ffn_dim"]),
        head_hidden=int(cfg["head_hidden"]),
        d_digit=int(cfg.get("d_digit", 16)),
        dropout=float(cfg["dropout"]),
        max_seq_len=int(cfg.get("max_seq_len", 256)),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    windows, labels, p = _build_trained_eval_dataset(args, cfg)
    variants = {
        variant: evaluate_attention_model(
            model,
            windows,
            labels,
            p=p,
            hierarchy_variant=variant,
            seed=int(cfg.get("seed", args.seed)),
            batch_size=args.trained_eval_batch_size,
            device=device,
        )
        for variant in ("true", "shuffled", "random")
    }
    return {
        "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
        "dataset": args.trained_eval_dataset,
        "device": str(device),
        "variants": variants,
    }


def main() -> None:
    args = parse_args()
    md_path = safe_results_path(args.output_md)
    device = resolve_device(args.device)

    if args.trained_eval_checkpoint:
        report = evaluate_trained_checkpoint(args, device)
        json_path = safe_results_path(args.trained_eval_output_json)
        md_eval_path = safe_results_path(args.trained_eval_output_md)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_trained_eval_markdown(md_eval_path, report)
        print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
        print(f"Wrote {md_eval_path.relative_to(REPO_ROOT)}")
        return

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
    write_markdown(md_path, report)
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the phase-1 PyTorch p-adic transformer benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from padic_transformer.config import BenchmarkConfig  # noqa: E402
from padic_transformer.hensel import digits_to_int64, int64_to_digits  # noqa: E402
from padic_transformer.ultrametric import (  # noqa: E402
    generate_clustered_hensel_dataset,
    nearest_center_accuracy,
    ultrametric_violation_rate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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

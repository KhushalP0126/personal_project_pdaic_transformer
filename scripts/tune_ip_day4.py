#!/usr/bin/env python3
"""Run the Day 4 CPU tuning pass for the IP-prefix experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from padic_transformer.report_paths import resolve_report_pair, safe_results_path


CONFIGS = [
    {
        "name": "baseline_w16_d64_l1_drop01_e1",
        "window_size": 16,
        "d_model": 64,
        "n_layers": 1,
        "dropout": 0.1,
        "epochs": 1,
    },
    {
        "name": "epochs3_w16_d64_l1_drop01",
        "window_size": 16,
        "d_model": 64,
        "n_layers": 1,
        "dropout": 0.1,
        "epochs": 3,
    },
    {
        "name": "d128_w16_l1_drop01_e1",
        "window_size": 16,
        "d_model": 128,
        "n_layers": 1,
        "dropout": 0.1,
        "epochs": 1,
    },
    {
        "name": "layers2_w16_d64_drop01_e1",
        "window_size": 16,
        "d_model": 64,
        "n_layers": 2,
        "dropout": 0.1,
        "epochs": 1,
    },
    {
        "name": "drop02_w16_d64_l1_e1",
        "window_size": 16,
        "d_model": 64,
        "n_layers": 1,
        "dropout": 0.2,
        "epochs": 1,
    },
    {
        "name": "window32_d64_l1_drop01_e1",
        "window_size": 32,
        "d_model": 64,
        "n_layers": 1,
        "dropout": 0.1,
        "epochs": 1,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--prefix-len", type=int, default=24)
    parser.add_argument("--num-prefixes", type=int, default=32)
    parser.add_argument("--attack-fraction", type=float, default=0.30)
    parser.add_argument("--attack-min-len", type=int, default=1)
    parser.add_argument("--attack-max-len", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-digit", type=int, default=8)
    parser.add_argument("--output-json", default="results/ip_day4_tuning.json")
    parser.add_argument("--output-md", default="results/ip_day4_tuning.md")
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


def run_config(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    json_path = safe_results_path(REPO_ROOT, f"results/ip_day4_{cfg['name']}.json")
    md_path = safe_results_path(REPO_ROOT, f"results/ip_day4_{cfg['name']}.md")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_ip_experiment.py"),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--train-samples",
        str(args.train_samples),
        "--val-samples",
        str(args.val_samples),
        "--window-size",
        str(cfg["window_size"]),
        "--prefix-len",
        str(args.prefix_len),
        "--num-prefixes",
        str(args.num_prefixes),
        "--attack-fraction",
        str(args.attack_fraction),
        "--attack-min-len",
        str(args.attack_min_len),
        "--attack-max-len",
        str(args.attack_max_len),
        "--epochs",
        str(cfg["epochs"]),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--d-model",
        str(cfg["d_model"]),
        "--n-heads",
        "4",
        "--n-layers",
        str(cfg["n_layers"]),
        "--d-digit",
        str(args.d_digit),
        "--dropout",
        str(cfg["dropout"]),
        "--output-json",
        str(json_path.relative_to(REPO_ROOT)),
        "--output-md",
        str(md_path.relative_to(REPO_ROOT)),
    ]
    print(f"\n=== Day 4 config: {cfg['name']} ===", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    return summarize_report(cfg, report, json_path, md_path)


def summarize_report(
    cfg: dict[str, Any],
    report: dict[str, Any],
    json_path: Path,
    md_path: Path,
) -> dict[str, Any]:
    models = report["models"]
    standard = models["standard_transformer"]
    hensel_only = models["hensel_only"]
    true = models["hensel_padic_sigmoid_true"]
    shuffled = models["hensel_padic_sigmoid_shuffled"]
    random = models["hensel_padic_sigmoid_random"]
    signed_alpha = models["hensel_padic_signed_alpha_true"]
    return {
        "name": cfg["name"],
        "config": cfg,
        "json": str(json_path.relative_to(REPO_ROOT)),
        "markdown": str(md_path.relative_to(REPO_ROOT)),
        "standard_auroc": standard["auroc"],
        "hensel_only_auroc": hensel_only["auroc"],
        "true_auroc": true["auroc"],
        "signed_alpha_auroc": signed_alpha["auroc"],
        "true_f1": true["f1"],
        "shuffled_auroc": shuffled["auroc"],
        "random_auroc": random["auroc"],
        "true_minus_hensel_only": true["auroc"] - hensel_only["auroc"],
        "true_minus_best_control": true["auroc"] - max(shuffled["auroc"], random["auroc"]),
        "signed_alpha_minus_hensel_only": signed_alpha["auroc"] - hensel_only["auroc"],
        "padic_alpha": true.get("padic_alpha"),
        "padic_attention_corr": true.get("padic_attention_corr"),
        "hierarchy_gap": true.get("hierarchy_gap"),
        "padic_alpha_grad_norm": true.get("padic_alpha_grad_norm"),
        "train_time_s": true.get("train_time_s"),
    }


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# IP Day 4 Tuning",
        "",
        "| Config | Standard | Hensel-only | Old gate true | Signed alpha | Shuffled | Random | Old-Hensel | Old-BestCtrl | Alpha | Alpha grad | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['name']} | {row['standard_auroc']:.4f} | {row['hensel_only_auroc']:.4f} | "
            f"{row['true_auroc']:.4f} | {row['signed_alpha_auroc']:.4f} | "
            f"{row['shuffled_auroc']:.4f} | {row['random_auroc']:.4f} | "
            f"{row['true_minus_hensel_only']:.4f} | {row['true_minus_best_control']:.4f} | "
            f"{row['padic_alpha']:.4f} | {row['padic_alpha_grad_norm']:.4f} | {row['train_time_s']:.2f} |"
        )
    best = max(summaries, key=lambda row: row["true_auroc"])
    best_clean = max(summaries, key=lambda row: row["true_minus_best_control"])
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Best old-gate AUROC: `{best['name']}` at `{best['true_auroc']:.4f}`.",
            f"- Best old-gate hierarchy-control gap: `{best_clean['name']}` at `{best_clean['true_minus_best_control']:.4f}`.",
            "",
            "Use the configuration that keeps the old-gate result above shuffled/random before comparing against hensel_only and signed alpha.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    summaries = [run_config(args, cfg) for cfg in CONFIGS]
    output_json, output_md = resolve_report_pair(
        REPO_ROOT,
        device,
        args.output_json,
        args.output_md,
        default_json="results/ip_day4_tuning.json",
        default_md="results/ip_day4_tuning.md",
    )
    output_json.write_text(json.dumps({"runs": summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_md, summaries)
    print(f"\nWrote {output_json.relative_to(REPO_ROOT)}")
    print(f"Wrote {output_md.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

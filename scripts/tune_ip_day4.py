#!/usr/bin/env python3
"""Run the Day 4 CPU tuning pass for the IP-prefix experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def result_path(raw_path: str) -> Path:
    path = (REPO_ROOT / raw_path).resolve()
    results_root = (REPO_ROOT / "results").resolve()
    if results_root not in (path, *path.parents):
        raise ValueError("outputs must be written under results/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_config(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    json_path = result_path(f"results/ip_day4_{cfg['name']}.json")
    md_path = result_path(f"results/ip_day4_{cfg['name']}.md")
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
    true = models["padic_attention_true"]
    shuffled = models["padic_attention_shuffled"]
    random = models["padic_attention_random"]
    vanilla = models["vanilla_transformer"]
    return {
        "name": cfg["name"],
        "config": cfg,
        "json": str(json_path.relative_to(REPO_ROOT)),
        "markdown": str(md_path.relative_to(REPO_ROOT)),
        "true_auroc": true["auroc"],
        "true_f1": true["f1"],
        "vanilla_auroc": vanilla["auroc"],
        "shuffled_auroc": shuffled["auroc"],
        "random_auroc": random["auroc"],
        "true_minus_vanilla": true["auroc"] - vanilla["auroc"],
        "true_minus_best_control": true["auroc"] - max(shuffled["auroc"], random["auroc"]),
        "padic_gate": true.get("padic_gate"),
        "padic_attention_corr": true.get("padic_attention_corr"),
        "hierarchy_gap": true.get("hierarchy_gap"),
        "train_time_s": true.get("train_time_s"),
    }


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# IP Day 4 Tuning",
        "",
        "| Config | True AUROC | Vanilla | Shuffled | Random | True - Vanilla | True - Best Control | Gate | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['name']} | {row['true_auroc']:.4f} | {row['vanilla_auroc']:.4f} | "
            f"{row['shuffled_auroc']:.4f} | {row['random_auroc']:.4f} | "
            f"{row['true_minus_vanilla']:.4f} | {row['true_minus_best_control']:.4f} | "
            f"{row['padic_gate']:.4f} | {row['train_time_s']:.2f} |"
        )
    best = max(summaries, key=lambda row: row["true_auroc"])
    best_clean = max(summaries, key=lambda row: row["true_minus_best_control"])
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Best true PDAIC AUROC: `{best['name']}` at `{best['true_auroc']:.4f}`.",
            f"- Best hierarchy-control gap: `{best_clean['name']}` at `{best_clean['true_minus_best_control']:.4f}`.",
            "",
            "Use the AUROC winner only if it still beats shuffled/random. Otherwise prefer the clean-control-gap winner for Day 5.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summaries = [run_config(args, cfg) for cfg in CONFIGS]
    output_json = result_path(args.output_json)
    output_md = result_path(args.output_md)
    output_json.write_text(json.dumps({"runs": summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_md, summaries)
    print(f"\nWrote {output_json.relative_to(REPO_ROOT)}")
    print(f"Wrote {output_md.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Day 5 multi-seed validation for the IP-prefix 2-adic experiment."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ORDER = (
    "logistic_regression",
    "isolation_forest",
    "vanilla_transformer",
    "padic_attention_true",
    "padic_attention_shuffled",
    "padic_attention_random",
)

METRICS = ("auroc", "f1", "precision", "recall", "accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260504, 20260505, 20260506])
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--prefix-len", type=int, default=24)
    parser.add_argument("--num-prefixes", type=int, default=32)
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
    parser.add_argument("--output-json", default="results/ip_day5_multiseed.json")
    parser.add_argument("--output-md", default="results/ip_day5_multiseed.md")
    return parser.parse_args()


def result_path(raw_path: str) -> Path:
    path = (REPO_ROOT / raw_path).resolve()
    results_root = (REPO_ROOT / "results").resolve()
    if results_root not in (path, *path.parents):
        raise ValueError("outputs must be written under results/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    json_path = result_path(f"results/ip_day5_seed_{seed}.json")
    md_path = result_path(f"results/ip_day5_seed_{seed}.md")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_ip_experiment.py"),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--train-samples",
        str(args.train_samples),
        "--val-samples",
        str(args.val_samples),
        "--window-size",
        str(args.window_size),
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
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--d-model",
        str(args.d_model),
        "--n-heads",
        str(args.n_heads),
        "--n-layers",
        str(args.n_layers),
        "--d-digit",
        str(args.d_digit),
        "--dropout",
        str(args.dropout),
        "--output-json",
        str(json_path.relative_to(REPO_ROOT)),
        "--output-md",
        str(md_path.relative_to(REPO_ROOT)),
    ]
    print(f"\n=== Day 5 seed {seed} ===", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "seed": seed,
        "json": str(json_path.relative_to(REPO_ROOT)),
        "markdown": str(md_path.relative_to(REPO_ROOT)),
        "report": report,
    }


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"models": {}, "gaps": {}, "attention": {}}

    for model_name in MODEL_ORDER:
        summary["models"][model_name] = {}
        for metric in METRICS:
            values = []
            for run in runs:
                model_metrics = run["report"]["models"].get(model_name, {})
                if metric in model_metrics:
                    values.append(float(model_metrics[metric]))
            summary["models"][model_name][metric] = mean_std(values)

    true_aurocs = [float(run["report"]["models"]["padic_attention_true"]["auroc"]) for run in runs]
    vanilla_aurocs = [float(run["report"]["models"]["vanilla_transformer"]["auroc"]) for run in runs]
    shuffled_aurocs = [float(run["report"]["models"]["padic_attention_shuffled"]["auroc"]) for run in runs]
    random_aurocs = [float(run["report"]["models"]["padic_attention_random"]["auroc"]) for run in runs]

    true_minus_vanilla = [t - v for t, v in zip(true_aurocs, vanilla_aurocs)]
    true_minus_shuffled = [t - s for t, s in zip(true_aurocs, shuffled_aurocs)]
    true_minus_random = [t - r for t, r in zip(true_aurocs, random_aurocs)]
    true_minus_best_control = [
        t - max(s, r) for t, s, r in zip(true_aurocs, shuffled_aurocs, random_aurocs)
    ]

    summary["gaps"] = {
        "true_minus_vanilla": mean_std(true_minus_vanilla),
        "true_minus_shuffled": mean_std(true_minus_shuffled),
        "true_minus_random": mean_std(true_minus_random),
        "true_minus_best_control": mean_std(true_minus_best_control),
        "wins_vs_vanilla": sum(g > 0 for g in true_minus_vanilla),
        "wins_vs_best_control": sum(g > 0 for g in true_minus_best_control),
        "num_seeds": len(runs),
    }

    gates = []
    corrs = []
    hierarchy_gaps = []
    for run in runs:
        true_metrics = run["report"]["models"]["padic_attention_true"]
        if "padic_gate" in true_metrics:
            gates.append(float(true_metrics["padic_gate"]))
        if "padic_attention_corr" in true_metrics:
            corrs.append(float(true_metrics["padic_attention_corr"]))
        if "hierarchy_gap" in true_metrics:
            hierarchy_gaps.append(float(true_metrics["hierarchy_gap"]))

    summary["attention"] = {
        "padic_gate": mean_std(gates),
        "padic_attention_corr": mean_std(corrs),
        "hierarchy_gap": mean_std(hierarchy_gaps),
    }
    return summary


def fmt_stat(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def write_markdown(path: Path, runs: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# IP Day 5 Multi-Seed Validation",
        "",
        "## Per-seed AUROC",
        "",
        "| Seed | Logistic | IsolationForest | Vanilla | True 2-adic | Shuffled | Random | True - Vanilla | True - Best Control |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for run in runs:
        seed = run["seed"]
        models = run["report"]["models"]
        logistic = models["logistic_regression"]["auroc"]
        iso = models["isolation_forest"]["auroc"]
        vanilla = models["vanilla_transformer"]["auroc"]
        true = models["padic_attention_true"]["auroc"]
        shuffled = models["padic_attention_shuffled"]["auroc"]
        random = models["padic_attention_random"]["auroc"]
        best_control = max(shuffled, random)
        lines.append(
            f"| {seed} | {logistic:.4f} | {iso:.4f} | {vanilla:.4f} | "
            f"{true:.4f} | {shuffled:.4f} | {random:.4f} | "
            f"{true - vanilla:.4f} | {true - best_control:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Mean ± std",
            "",
            "| Model | AUROC | F1 | Precision | Recall | Accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for model_name in MODEL_ORDER:
        row = summary["models"][model_name]
        lines.append(
            f"| {model_name} | "
            f"{fmt_stat(row['auroc'])} | "
            f"{fmt_stat(row['f1'])} | "
            f"{fmt_stat(row['precision'])} | "
            f"{fmt_stat(row['recall'])} | "
            f"{fmt_stat(row['accuracy'])} |"
        )

    gaps = summary["gaps"]
    attention = summary["attention"]
    lines.extend(
        [
            "",
            "## Paper-critical gaps",
            "",
            "| Comparison | Mean ± std |",
            "|---|---:|",
            f"| True 2-adic - vanilla | {fmt_stat(gaps['true_minus_vanilla'])} |",
            f"| True 2-adic - shuffled | {fmt_stat(gaps['true_minus_shuffled'])} |",
            f"| True 2-adic - random | {fmt_stat(gaps['true_minus_random'])} |",
            f"| True 2-adic - best control | {fmt_stat(gaps['true_minus_best_control'])} |",
            "",
            "## Win counts",
            "",
            f"- True 2-adic beats vanilla in `{gaps['wins_vs_vanilla']}/{gaps['num_seeds']}` seeds.",
            f"- True 2-adic beats best hierarchy control in `{gaps['wins_vs_best_control']}/{gaps['num_seeds']}` seeds.",
            "",
            "## Attention diagnostics",
            "",
            "| Metric | Mean ± std |",
            "|---|---:|",
            f"| p-adic gate | {fmt_stat(attention['padic_gate'])} |",
            f"| p-adic attention corr | {fmt_stat(attention['padic_attention_corr'])} |",
            f"| hierarchy gap | {fmt_stat(attention['hierarchy_gap'])} |",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = [run_seed(args, seed) for seed in args.seeds]
    summary = summarize(runs)
    output_json = result_path(args.output_json)
    output_md = result_path(args.output_md)
    payload = {
        "config": {
            "seeds": args.seeds,
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "window_size": args.window_size,
            "prefix_len": args.prefix_len,
            "num_prefixes": args.num_prefixes,
            "attack_fraction": args.attack_fraction,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_digit": args.d_digit,
            "dropout": args.dropout,
        },
        "runs": [
            {
                "seed": run["seed"],
                "json": run["json"],
                "markdown": run["markdown"],
            }
            for run in runs
        ],
        "summary": summary,
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_md, runs, summary)
    print(f"\nWrote {output_json.relative_to(REPO_ROOT)}")
    print(f"Wrote {output_md.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

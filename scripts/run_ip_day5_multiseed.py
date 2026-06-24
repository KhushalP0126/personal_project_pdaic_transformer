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

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from padic_transformer.report_paths import resolve_report_pair, safe_results_path

MODEL_ORDER = (
    "logistic_regression",
    "isolation_forest",
    "standard_transformer",
    "hensel_only",
    "hensel_padic_sigmoid_true",
    "hensel_padic_sigmoid_shuffled",
    "hensel_padic_sigmoid_random",
    "hensel_padic_signed_alpha_true",
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
    parser.add_argument("--gate-init-logit", type=float, default=0.0)
    parser.add_argument("--gate-regularization-weight", type=float, default=0.001)
    parser.add_argument("--fixed-padic-gate", type=float, default=None)
    parser.add_argument("--output-json", default="results/ip_day5_multiseed.json")
    parser.add_argument("--output-md", default="results/ip_day5_multiseed.md")
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


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    json_path = safe_results_path(REPO_ROOT, f"results/ip_day5_seed_{seed}.json")
    md_path = safe_results_path(REPO_ROOT, f"results/ip_day5_seed_{seed}.md")
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
        "--gate-init-logit",
        str(args.gate_init_logit),
        "--gate-regularization-weight",
        str(args.gate_regularization_weight),
        "--output-json",
        str(json_path.relative_to(REPO_ROOT)),
        "--output-md",
        str(md_path.relative_to(REPO_ROOT)),
    ]
    if args.fixed_padic_gate is not None:
        command.extend(["--fixed-padic-gate", str(args.fixed_padic_gate)])
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

    true_aurocs = [float(run["report"]["models"]["hensel_padic_sigmoid_true"]["auroc"]) for run in runs]
    standard_aurocs = [float(run["report"]["models"]["standard_transformer"]["auroc"]) for run in runs]
    hensel_only_aurocs = [float(run["report"]["models"]["hensel_only"]["auroc"]) for run in runs]
    shuffled_aurocs = [float(run["report"]["models"]["hensel_padic_sigmoid_shuffled"]["auroc"]) for run in runs]
    random_aurocs = [float(run["report"]["models"]["hensel_padic_sigmoid_random"]["auroc"]) for run in runs]
    signed_alpha_aurocs = [float(run["report"]["models"]["hensel_padic_signed_alpha_true"]["auroc"]) for run in runs]

    true_minus_standard = [t - v for t, v in zip(true_aurocs, standard_aurocs)]
    true_minus_hensel_only = [t - v for t, v in zip(true_aurocs, hensel_only_aurocs)]
    true_minus_shuffled = [t - s for t, s in zip(true_aurocs, shuffled_aurocs)]
    true_minus_random = [t - r for t, r in zip(true_aurocs, random_aurocs)]
    true_minus_best_control = [
        t - max(s, r) for t, s, r in zip(true_aurocs, shuffled_aurocs, random_aurocs)
    ]
    signed_alpha_minus_hensel_only = [t - v for t, v in zip(signed_alpha_aurocs, hensel_only_aurocs)]

    summary["gaps"] = {
        "true_minus_standard": mean_std(true_minus_standard),
        "true_minus_hensel_only": mean_std(true_minus_hensel_only),
        "true_minus_shuffled": mean_std(true_minus_shuffled),
        "true_minus_random": mean_std(true_minus_random),
        "true_minus_best_control": mean_std(true_minus_best_control),
        "signed_alpha_minus_hensel_only": mean_std(signed_alpha_minus_hensel_only),
        "wins_vs_standard": sum(g > 0 for g in true_minus_standard),
        "wins_vs_hensel_only": sum(g > 0 for g in true_minus_hensel_only),
        "wins_vs_best_control": sum(g > 0 for g in true_minus_best_control),
        "num_seeds": len(runs),
    }

    gates = []
    corrs = []
    hierarchy_gaps = []
    for run in runs:
        true_metrics = run["report"]["models"]["hensel_padic_sigmoid_true"]
        if "padic_alpha" in true_metrics:
            gates.append(float(true_metrics["padic_alpha"]))
        if "padic_alpha_grad_norm" in true_metrics:
            summary["attention"].setdefault("padic_alpha_grad_norm_values", []).append(
                float(true_metrics["padic_alpha_grad_norm"])
            )
        if "padic_attention_corr" in true_metrics:
            corrs.append(float(true_metrics["padic_attention_corr"]))
        if "hierarchy_gap" in true_metrics:
            hierarchy_gaps.append(float(true_metrics["hierarchy_gap"]))

    summary["attention"] = {
        "padic_alpha": mean_std(gates),
        "padic_attention_corr": mean_std(corrs),
        "hierarchy_gap": mean_std(hierarchy_gaps),
        "padic_alpha_grad_norm": mean_std(summary["attention"].pop("padic_alpha_grad_norm_values", [])),
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
        "| Seed | Standard | Hensel-only | Old gate | Signed alpha | Shuffled | Random | Old-Hensel | Old-Best Control |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for run in runs:
        seed = run["seed"]
        models = run["report"]["models"]
        standard = models["standard_transformer"]["auroc"]
        hensel_only = models["hensel_only"]["auroc"]
        true = models["hensel_padic_sigmoid_true"]["auroc"]
        signed_alpha = models["hensel_padic_signed_alpha_true"]["auroc"]
        shuffled = models["hensel_padic_sigmoid_shuffled"]["auroc"]
        random = models["hensel_padic_sigmoid_random"]["auroc"]
        best_control = max(shuffled, random)
        lines.append(
            f"| {seed} | {standard:.4f} | {hensel_only:.4f} | {true:.4f} | "
            f"{signed_alpha:.4f} | {shuffled:.4f} | {random:.4f} | "
            f"{true - hensel_only:.4f} | {true - best_control:.4f} |"
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
            f"| Old gate - standard | {fmt_stat(gaps['true_minus_standard'])} |",
            f"| Old gate - hensel-only | {fmt_stat(gaps['true_minus_hensel_only'])} |",
            f"| True 2-adic - shuffled | {fmt_stat(gaps['true_minus_shuffled'])} |",
            f"| True 2-adic - random | {fmt_stat(gaps['true_minus_random'])} |",
            f"| True 2-adic - best control | {fmt_stat(gaps['true_minus_best_control'])} |",
            f"| Signed alpha - hensel-only | {fmt_stat(gaps['signed_alpha_minus_hensel_only'])} |",
            "",
            "## Win counts",
            "",
            f"- Old gate beats standard in `{gaps['wins_vs_standard']}/{gaps['num_seeds']}` seeds.",
            f"- Old gate beats hensel-only in `{gaps['wins_vs_hensel_only']}/{gaps['num_seeds']}` seeds.",
            f"- True 2-adic beats best hierarchy control in `{gaps['wins_vs_best_control']}/{gaps['num_seeds']}` seeds.",
            "",
            "## Attention diagnostics",
            "",
            "| Metric | Mean ± std |",
            "|---|---:|",
            f"| p-adic alpha | {fmt_stat(attention['padic_alpha'])} |",
            f"| p-adic alpha grad norm | {fmt_stat(attention['padic_alpha_grad_norm'])} |",
            f"| p-adic attention corr | {fmt_stat(attention['padic_attention_corr'])} |",
            f"| hierarchy gap | {fmt_stat(attention['hierarchy_gap'])} |",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    runs = [run_seed(args, seed) for seed in args.seeds]
    summary = summarize(runs)
    output_json, output_md = resolve_report_pair(
        REPO_ROOT,
        device,
        args.output_json,
        args.output_md,
        default_json="results/ip_day5_multiseed.json",
        default_md="results/ip_day5_multiseed.md",
    )
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
            "gate_init_logit": args.gate_init_logit,
            "gate_regularization_weight": args.gate_regularization_weight,
            "fixed_padic_gate": args.fixed_padic_gate,
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

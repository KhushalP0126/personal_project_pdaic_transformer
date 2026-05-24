#!/usr/bin/env python3
"""Compare vanilla and PDAIC prime-sweep training logs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def safe_results_path(raw_path: str) -> Path:
    path = (REPO_ROOT / raw_path).resolve()
    results_root = (REPO_ROOT / "results").resolve()
    if results_root not in (path, *path.parents):
        raise ValueError("outputs must be written under results/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_result(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"missing training log: {path.relative_to(REPO_ROOT)}. "
            "Run `make primes` and `make pdaic-primes` first, or run `make analysis`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def metric(result: dict, key: str) -> float:
    value = result.get(key, float("nan"))
    return float(value) if value is not None else float("nan")


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-list", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--vanilla-template", default="results/compare_p{p}.json")
    parser.add_argument("--pdaic-template", default="results/compare_pdaic_p{p}.json")
    parser.add_argument("--output-json", default="results/prime_comparison.json")
    parser.add_argument("--output-md", default="results/prime_comparison.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for p in args.p_list:
        vanilla = load_result(REPO_ROOT / args.vanilla_template.format(p=p))
        pdaic = load_result(REPO_ROOT / args.pdaic_template.format(p=p))
        vanilla_auroc = metric(vanilla, "best_auroc")
        pdaic_auroc = metric(pdaic, "best_auroc")
        rows.append(
            {
                "p": p,
                "vanilla_auroc": vanilla_auroc,
                "pdaic_auroc": pdaic_auroc,
                "auroc_delta": pdaic_auroc - vanilla_auroc,
                "vanilla_f1": metric(vanilla, "best_f1"),
                "pdaic_f1": metric(pdaic, "best_f1"),
                "f1_delta": metric(pdaic, "best_f1") - metric(vanilla, "best_f1"),
                "vanilla_epoch": int(vanilla.get("best_epoch", 0)),
                "pdaic_epoch": int(pdaic.get("best_epoch", 0)),
                "vanilla_seconds": metric(vanilla, "total_seconds"),
                "pdaic_seconds": metric(pdaic, "total_seconds"),
            }
        )

    output_json = safe_results_path(args.output_json)
    output_md = safe_results_path(args.output_md)
    output_json.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Prime Sweep Comparison",
        "",
        "| p | Vanilla AUROC | PDAIC AUROC | Delta | Vanilla F1 | PDAIC F1 | Delta | Vanilla epoch | PDAIC epoch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['p']} | {fmt(row['vanilla_auroc'])} | {fmt(row['pdaic_auroc'])} | "
            f"{fmt(row['auroc_delta'])} | {fmt(row['vanilla_f1'])} | {fmt(row['pdaic_f1'])} | "
            f"{fmt(row['f1_delta'])} | {row['vanilla_epoch']} | {row['pdaic_epoch']} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_json.relative_to(REPO_ROOT)}")
    print(f"Wrote {output_md.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

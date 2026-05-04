#!/usr/bin/env python3
"""Verify unsigned INT8 arithmetic against truncated 2-adic arithmetic."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from padic_transformer.int8_2adic import (  # noqa: E402
    DEFAULT_SYSCALL_MAP,
    matmul2x2_mod256,
    syscall_distance_rows,
    verify_numpy_int8,
    wrap_uint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--output-json", default="results/int8_2adic_report.json")
    parser.add_argument("--output-md", default="results/int8_2adic_report.md")
    return parser.parse_args()


def safe_results_path(raw_path: str) -> Path:
    path = (REPO_ROOT / raw_path).resolve()
    results_root = (REPO_ROOT / "results").resolve()
    if results_root not in (path, *path.parents):
        raise ValueError("outputs must be written under results/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_markdown(path: Path, report: dict[str, object]) -> None:
    verification = report["verification"]
    rows = [
        f"| Pair count | {verification['pair_count']} |",
        f"| Unsigned INT8 mismatches | {verification['unsigned_mismatches']} |",
        f"| Signed low-byte mismatches | {verification['signed_low_byte_mismatches']} |",
        f"| Signed wide low-byte mismatches | {verification['signed_wide_low_byte_mismatches']} |",
        f"| Signed integer semantic mismatches | {verification['signed_integer_semantic_mismatches']} |",
        f"| 2-adic valuation mismatches | {verification['valuation_mismatches']} |",
        f"| Hensel multiplication mismatches | {verification['hensel_mul_mismatches']} |",
    ]
    syscall_rows = [
        f"| `{left}` | `{right}` | {exponent} | `2^-{exponent}` |"
        for left, right, exponent in report["syscall_distances"]
    ]
    matrix_rows = [
        f"| {row_idx} | {row[0]} | {row[1]} |"
        for row_idx, row in enumerate(report["matmul2x2_mod256"])
    ]

    body = "\n".join(
        [
            "# INT8 2-adic Hardware Dry-Lab",
            "",
            f"- Generated UTC: `{report['generated_utc']}`",
            f"- Precision: `r={verification['precision']}`",
            f"- Modulus: `2^{verification['precision']} = 256`",
            "",
            "## Exhaustive Multiply Check",
            "",
            "| Check | Count |",
            "|---|---:|",
            *rows,
            "",
            "Unsigned INT8 wraparound matches truncated 2-adic multiplication when the low 8 bits are retained.",
            "Signed low-byte products still match modulo 256, but signed integer semantics do not match unsigned 2-adic residues.",
            "The tile must avoid signed interpretation in custom distance and threshold logic.",
            "",
            "## Syscall Distance Sanity Map",
            "",
            "| Left | Right | shared low-bit exponent | distance |",
            "|---|---|---:|---|",
            *syscall_rows,
            "",
            "The sample mapping puts behaviorally related calls on the same low-order 2-adic branch.",
            "",
            "## 2x2 Dot Product Mod 256",
            "",
            f"Left matrix: `{report['matmul_left']}`",
            f"Right matrix: `{report['matmul_right']}`",
            "",
            "| Row | Col 0 | Col 1 |",
            "|---:|---:|---:|",
            *matrix_rows,
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    json_path = safe_results_path(args.output_json)
    md_path = safe_results_path(args.output_md)

    verification = verify_numpy_int8(args.r)
    left = [[0x03, 0x05], [0x80, 0xFF]]
    right = [[0x07, 0x02], [0x04, 0x09]]
    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verification": verification.__dict__,
        "syscall_map": DEFAULT_SYSCALL_MAP,
        "syscall_distances": syscall_distance_rows(DEFAULT_SYSCALL_MAP),
        "matmul_left": left,
        "matmul_right": right,
        "matmul2x2_mod256": matmul2x2_mod256(left, right),
        "minus_one_uint8": wrap_uint(-1, 8),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

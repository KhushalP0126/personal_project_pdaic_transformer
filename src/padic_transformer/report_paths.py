"""Helpers for organizing results paths across CPU and GPU runs."""

from __future__ import annotations

from pathlib import Path

import torch


REPORT_SLOT_NAMES = ("report", "report2")


def safe_results_path(repo_root: Path, raw_path: str) -> Path:
    path = (repo_root / raw_path).resolve()
    results_root = (repo_root / "results").resolve()
    if results_root not in (path, *path.parents):
        raise ValueError("outputs must be written under results/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _slot_timestamp(results_root: Path, slot: str) -> float:
    timestamps = []
    for suffix in (".json", ".md"):
        path = results_root / f"{slot}{suffix}"
        if path.exists():
            timestamps.append(path.stat().st_mtime)
    return max(timestamps) if timestamps else -1.0


def choose_report_slot(results_root: Path) -> str:
    slot_paths = {
        slot: [results_root / f"{slot}.json", results_root / f"{slot}.md"]
        for slot in REPORT_SLOT_NAMES
    }
    for slot in REPORT_SLOT_NAMES:
        if not any(path.exists() for path in slot_paths[slot]):
            return slot

    ordered = sorted(
        REPORT_SLOT_NAMES,
        key=lambda slot: (_slot_timestamp(results_root, slot), slot),
    )
    return ordered[0]


def resolve_report_pair(
    repo_root: Path,
    device: torch.device,
    raw_json_path: str,
    raw_md_path: str,
    *,
    default_json: str,
    default_md: str,
) -> tuple[Path, Path]:
    if (
        device.type == "cpu"
        and raw_json_path == default_json
        and raw_md_path == default_md
    ):
        results_root = (repo_root / "results").resolve()
        slot = choose_report_slot(results_root)
        return (
            safe_results_path(repo_root, f"results/{slot}.json"),
            safe_results_path(repo_root, f"results/{slot}.md"),
        )
    return (
        safe_results_path(repo_root, raw_json_path),
        safe_results_path(repo_root, raw_md_path),
    )


def resolve_report_json(
    repo_root: Path,
    device: torch.device,
    raw_json_path: str,
    *,
    default_json: str,
) -> Path:
    if device.type == "cpu" and raw_json_path == default_json:
        results_root = (repo_root / "results").resolve()
        slot = choose_report_slot(results_root)
        return safe_results_path(repo_root, f"results/{slot}.json")
    return safe_results_path(repo_root, raw_json_path)

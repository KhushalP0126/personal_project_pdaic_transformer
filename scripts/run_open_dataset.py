#!/usr/bin/env python3
"""Download, parse, validate, and benchmark open anomaly datasets."""

from __future__ import annotations

import argparse
import json
import csv
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent if (HERE.parent / "src").exists() else HERE
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from padic_transformer.baselines_and_validation import (
    run_isolation_forest_baseline,
    run_standard_transformer_baseline,
)
from padic_transformer.hensel import int64_to_digits
from padic_transformer.metrics import binary_auroc
from padic_transformer.model import PadicAnomalyDetector
from padic_transformer.model_fixes import StreamingWindowScorer, quantize_dynamic_model
from padic_transformer.ultrametric import ultrametric_violation_rate

ADFA_REPO_URL = "https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset.git"
ADFA_SYSCALL_VOCAB_SIZE = 200


def safe_results_path(raw_path: str) -> Path:
    path = (REPO / raw_path).resolve()
    results_root = (REPO / "results").resolve()
    if results_root not in (path, *path.parents):
        raise ValueError("outputs must be written under results/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def download_adfa(data_dir: Path) -> Path:
    repo_dir = data_dir / "a-labelled-version-of-the-ADFA-LD-dataset"
    if repo_dir.exists():
        return repo_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", ADFA_REPO_URL, str(repo_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed:\n{result.stderr}\n"
            "Make sure git is installed or download manually."
        )

    zip_path = repo_dir / "ADFA-LD.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(repo_dir)
    return repo_dir


def download_beth(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list(data_dir.glob("labelled_*.csv")) or list(data_dir.glob("*.csv"))
    if csv_files:
        return data_dir

    result = subprocess.run(
        ["kaggle", "datasets", "download", "katehighnam/beth-dataset", "-p", str(data_dir), "--unzip"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle download failed:\n{result.stderr}\n"
            "Install kaggle CLI and configure credentials, or download manually."
        )
    return data_dir


def _read_trace_file(filepath: Path) -> list[int]:
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    return [int(x) for x in text.split() if x.strip().isdigit()]


def _build_vocab(all_traces: list[list[int]], max_vocab: int) -> dict[int, int]:
    counter: Counter[int] = Counter()
    for trace in all_traces:
        counter.update(trace)
    top = [sid for sid, _ in counter.most_common(max_vocab)]
    return {sid: idx for idx, sid in enumerate(top)}


def load_adfa_ld(
    data_dir: Path,
    p: int = 3,
    r: int = 8,
    window_size: int = 32,
    stride: int = 4,
    max_vocab: int = ADFA_SYSCALL_VOCAB_SIZE,
    download: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object], list[str]]:
    if download:
        repo_dir = download_adfa(data_dir)
    else:
        repo_dir = data_dir

    adfa_root = repo_dir / "ADFA-LD"
    if not adfa_root.exists():
        adfa_root = repo_dir
    if not adfa_root.exists():
        raise FileNotFoundError(f"Could not find ADFA-LD directory under {repo_dir}")

    raw_traces: list[tuple[list[int], int, str]] = []
    for split_name, label in [("Training_Data_Master", 0), ("Validation_Data_Master", 0)]:
        split_dir = adfa_root / split_name
        if not split_dir.exists():
            continue
        for fp in sorted(split_dir.glob("*.txt")):
            ids = _read_trace_file(fp)
            if ids:
                raw_traces.append((ids, label, split_name))

    attack_dir = adfa_root / "Attack_Data_Master"
    if attack_dir.exists():
        for family_dir in sorted(attack_dir.iterdir()):
            if not family_dir.is_dir():
                continue
            for fp in sorted(family_dir.glob("*.txt")):
                ids = _read_trace_file(fp)
                if ids:
                    raw_traces.append((ids, 1, family_dir.name))

    if not raw_traces:
        raise RuntimeError(f"No trace files found under {adfa_root}")

    normal_traces = [ids for ids, lbl, _ in raw_traces if lbl == 0]
    vocab = _build_vocab(normal_traces, max_vocab)
    oov_idx = max_vocab
    vocab_size = max_vocab + 1

    all_windows: list[torch.Tensor] = []
    all_labels: list[float] = []
    all_families: list[str] = []
    trace_lengths: list[int] = []

    for trace_ids, label, _source in raw_traces:
        compact = [vocab.get(sid, oov_idx) for sid in trace_ids]
        trace_lengths.append(len(compact))
        if len(compact) < window_size:
            compact = compact + [oov_idx] * (window_size - len(compact))

        id_tensor = torch.tensor(compact, dtype=torch.int64)
        id_clamped = id_tensor.clamp(0, p**r - 1)
        digit_seq = int64_to_digits(id_clamped, p=p, r=r)
        n_windows = max(1, (len(compact) - window_size) // stride + 1)
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            if end > digit_seq.shape[0]:
                break
            all_windows.append(digit_seq[start:end].clone())
            all_labels.append(float(label))
            all_families.append("normal" if label == 0 else str(_source))

    windows_tensor = torch.stack(all_windows)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)

    n_pos = int(labels_tensor.sum().item())
    n_neg = len(labels_tensor) - n_pos
    stats: dict[str, object] = {
        "n_windows": len(labels_tensor),
        "n_normal": n_neg,
        "n_attack": n_pos,
        "real_attack_rate": n_pos / max(1, len(labels_tensor)),
        "pos_weight": n_neg / max(1, n_pos),
        "vocab_size": vocab_size,
        "mean_trace_len": sum(trace_lengths) / max(1, len(trace_lengths)),
        "min_trace_len": min(trace_lengths),
        "max_trace_len": max(trace_lengths),
        "p": p,
        "r": r,
        "window_size": window_size,
        "stride": stride,
    }
    stats["attack_family_counts"] = {
        family: all_families.count(family)
        for family in sorted({family for family in all_families if family != "normal"})
    }
    return windows_tensor, labels_tensor, stats, all_families


def _read_beth_csvs(data_path: Path, max_rows: int) -> list[dict[str, str]]:
    csv_files = list(data_path.glob("labelled_*.csv")) or list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_path}. Download BETH from Kaggle first."
        )

    rows: list[dict[str, str]] = []
    for csv_file in sorted(csv_files):
        with open(csv_file, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
                if len(rows) >= max_rows:
                    return rows
    return rows


def load_beth(
    data_dir: Path,
    p: int = 3,
    r: int = 8,
    window_size: int = 32,
    stride: int = 4,
    max_rows: int = 500_000,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object], list[str]]:
    rows = _read_beth_csvs(data_dir, max_rows=max_rows)
    pid_sequences: dict[int, list[tuple[int, int]]] = {}
    for row in rows:
        try:
            pid = int(row.get("processId", 0) or 0)
            event_id = int(row.get("eventId", 0) or 0)
            evil = int(row.get("evil", row.get("sus", 0)) or 0)
        except (ValueError, KeyError):
            continue
        pid_sequences.setdefault(pid, []).append((event_id, evil))

    if not pid_sequences:
        raise RuntimeError("No usable rows found in BETH CSV files")

    all_event_ids = [eid for seq in pid_sequences.values() for eid, _ in seq]
    counter: Counter[int] = Counter(all_event_ids)
    top_events = [eid for eid, _ in counter.most_common(199)]
    vocab = {eid: idx for idx, eid in enumerate(top_events)}
    oov_idx = 199

    all_windows: list[torch.Tensor] = []
    all_labels: list[float] = []
    all_families: list[str] = []
    for seq in pid_sequences.values():
        event_ids = [vocab.get(eid, oov_idx) for eid, _ in seq]
        evil_labels = [lbl for _, lbl in seq]
        if len(event_ids) < window_size:
            continue
        id_tensor = torch.tensor(event_ids, dtype=torch.int64).clamp(0, p**r - 1)
        lbl_tensor = torch.tensor(evil_labels, dtype=torch.float32)
        digit_seq = int64_to_digits(id_tensor, p=p, r=r)
        for start in range(0, len(event_ids) - window_size + 1, stride):
            end = start + window_size
            window = digit_seq[start:end]
            window_label = float(lbl_tensor[start:end].max().item())
            all_windows.append(window.clone())
            all_labels.append(window_label)
            all_families.append("attack" if window_label > 0 else "normal")

    if not all_windows:
        raise RuntimeError("No windows could be built from BETH data")

    windows_tensor = torch.stack(all_windows)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
    n_pos = int(labels_tensor.sum().item())
    n_neg = len(labels_tensor) - n_pos
    stats: dict[str, object] = {
        "n_windows": len(labels_tensor),
        "n_normal": n_neg,
        "n_attack": n_pos,
        "real_attack_rate": n_pos / max(1, len(labels_tensor)),
        "pos_weight": n_neg / max(1, n_pos),
        "p": p,
        "r": r,
        "window_size": window_size,
        "stride": stride,
    }
    return windows_tensor, labels_tensor, stats, all_families


def check_ultrametric(windows: torch.Tensor, labels: torch.Tensor, n_triplets: int = 5000) -> dict[str, object]:
    normal_mask = labels == 0
    normal_windows = windows[normal_mask]
    if normal_windows.shape[0] < 3:
        return {"error": "Not enough normal windows for triplet sampling"}
    flat_tokens = normal_windows.reshape(-1, normal_windows.shape[-1])
    n = min(flat_tokens.shape[0], 10_000)
    idx = torch.randperm(flat_tokens.shape[0])[:n]
    sample = flat_tokens[idx]
    t0 = time.perf_counter()
    rate, count = ultrametric_violation_rate(sample, triplets=n_triplets, seed=20260504)
    elapsed = time.perf_counter() - t0
    verdict = (
        "GOOD — p-adic structure is plausible on this data"
        if rate < 0.05
        else "MODERATE — some hierarchy, but noisy"
        if rate < 0.20
        else "POOR — ultrametric assumption likely does not hold"
    )
    return {
        "violation_rate": rate,
        "violations": count,
        "triplets": n_triplets,
        "tokens_checked": n,
        "elapsed_s": elapsed,
        "verdict": verdict,
    }


def train_and_eval(
    windows: torch.Tensor,
    labels: torch.Tensor,
    stats: dict[str, object],
    families: list[str] | None,
    p: int,
    r: int,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    epochs: int = 5,
    batch_size: int = 256,
    device_str: str = "cpu",
    run_isolation_forest: bool = True,
    quantize_int8: bool = False,
    seed: int = 20260504,
) -> dict[str, object]:
    from torch.utils.data import DataLoader, TensorDataset, random_split

    from padic_transformer.losses import AnomalyLoss

    torch.manual_seed(seed)
    device = torch.device(device_str)
    n = len(labels)
    n_train = int(0.8 * n)
    n_val = n - n_train
    ds = TensorDataset(windows, labels)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    pos_weight = float(stats["pos_weight"])
    results: dict[str, object] = {}

    if run_isolation_forest:
        iso = run_isolation_forest_baseline(
            windows[train_ds.indices],
            labels[train_ds.indices],
            windows[val_ds.indices],
            labels[val_ds.indices],
        )
        results["isolation_forest"] = iso

    model = PadicAnomalyDetector(
        p=p,
        r=r,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=d_model * 4,
        head_hidden=d_model // 2,
    ).to(device)
    loss_fn = AnomalyLoss(p=p, alpha=0.5, pos_weight=pos_weight).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    train_loader = DataLoader(
        TensorDataset(windows[train_ds.indices], labels[train_ds.indices]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(windows[val_ds.indices], labels[val_ds.indices]),
        batch_size=batch_size * 2,
        shuffle=False,
    )

    best_auroc = 0.0
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for win_batch, lbl_batch in train_loader:
            win_batch = win_batch.to(device)
            lbl_batch = lbl_batch.to(device)
            logits, reps = model.forward_with_features(win_batch)
            loss, _, _ = loss_fn(logits, lbl_batch, reps)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for win_batch, lbl_batch in val_loader:
                logits, _ = model.forward_with_features(win_batch.to(device))
                all_logits.append(logits.cpu())
                all_labels.append(lbl_batch.cpu())

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels).long()
        auroc = binary_auroc(logits_cat, labels_cat)
        best_auroc = max(best_auroc, auroc)

    results["padic_transformer"] = {"auroc": best_auroc, "time_s": time.perf_counter() - t0}

    if families is not None:
        model.eval()
        family_summary: dict[str, dict[str, float]] = {}
        with torch.no_grad():
            logits = model(windows.to(device)).cpu()
        for family in sorted(set(families)):
            if family == "normal":
                continue
            mask = torch.tensor([f == family for f in families], dtype=torch.bool)
            if mask.any():
                family_logits = logits[mask]
                family_summary[family] = {
                    "count": float(mask.sum().item()),
                    "mean_logit": float(family_logits.mean().item()),
                    "attack_rate_at_zero": float((family_logits >= 0).float().mean().item()),
                }
        results["attack_family_summary"] = family_summary

    if quantize_int8 and device.type == "cpu":
        q_model = quantize_dynamic_model(model.cpu())
        q_model.eval()
        sample_batch = windows[: min(batch_size, len(windows))]
        t_q = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                _ = q_model(sample_batch)
        q_latency = (time.perf_counter() - t_q) / 20.0
        t_fp = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                _ = model(sample_batch)
        fp_latency = (time.perf_counter() - t_fp) / 20.0
        results["int8_cpu_benchmark"] = {
            "fp32_latency_s": fp_latency,
            "int8_latency_s": q_latency,
            "speedup": fp_latency / max(q_latency, 1e-9),
        }

    # Streaming benchmark on the validation split representative batch.
    scorer = StreamingWindowScorer(model, window_size=windows.shape[1])
    stream_tokens = windows[val_ds.indices[0]].to(device)
    t_stream = time.perf_counter()
    with torch.no_grad():
        logits = scorer.push(stream_tokens)
    results["streaming_inference"] = {
        "final_logit": None if logits is None else float(logits.item()),
        "elapsed_s": time.perf_counter() - t_stream,
    }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["adfa", "beth"], default="adfa")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--triplets", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=3, help="Number of stability runs to average")
    parser.add_argument("--quantize-int8", action="store_true", help="Benchmark a dynamically quantized CPU model")
    return parser.parse_args()


def _write_markdown(path: Path, dataset: str, stats: dict[str, object], um: dict[str, object], results: dict[str, object] | None) -> None:
    rows = [
        f"- Dataset: `{dataset}`",
        f"- Windows: `{stats['n_windows']}`",
        f"- Normal: `{stats['n_normal']}`",
        f"- Attack: `{stats['n_attack']}`",
        f"- Real attack rate: `{float(stats['real_attack_rate']) * 100:.4f}%`",
        f"- pos_weight: `{float(stats['pos_weight']):.4f}`",
        f"- Ultrametric violation rate: `{float(um['violation_rate']):.4f}`",
        f"- Ultrametric verdict: `{um['verdict']}`",
    ]
    if results:
        if "multi_seed" in results:
            rows.append(
                f"- Multi-seed AUROC mean/std: `{results['multi_seed']['auroc_mean']:.4f}` / `{results['multi_seed']['auroc_std']:.4f}`"
            )
        if "isolation_forest" in results:
            rows.append(
                f"- IsolationForest AUROC: `{results['isolation_forest']['auroc']:.4f}`"
            )
        rows.append(
            f"- p-adic transformer AUROC: `{results['padic_transformer']['auroc']:.4f}`"
        )
        if "int8_cpu_benchmark" in results:
            rows.append(
                f"- INT8 speedup: `{results['int8_cpu_benchmark']['speedup']:.2f}x`"
            )
        if "streaming_inference" in results:
            rows.append(
                f"- Streaming final logit: `{results['streaming_inference']['final_logit']}`"
            )
        if "attack_family_summary" in results:
            for family, fam_stats in results["attack_family_summary"].items():
                rows.append(
                    f"- Family `{family}`: count `{int(fam_stats['count'])}`, mean logit `{fam_stats['mean_logit']:.4f}`, attack@0 `{fam_stats['attack_rate_at_zero']:.4f}`"
                )
    path.write_text("# Open Dataset Run\n\n" + "\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.dataset == "adfa":
        windows, labels, stats, families = load_adfa_ld(
            data_dir,
            p=args.p,
            r=args.r,
            window_size=args.window_size,
            stride=args.stride,
            download=not args.no_download,
        )
    else:
        if not args.no_download:
            download_beth(data_dir)
        windows, labels, stats, families = load_beth(
            data_dir,
            p=args.p,
            r=args.r,
            window_size=args.window_size,
            stride=args.stride,
        )

    um_results = check_ultrametric(windows, labels, n_triplets=args.triplets)

    out_json = safe_results_path("results/open_dataset_report.json")
    out_md = safe_results_path("results/open_dataset_report.md")
    results: dict[str, object] | None = None
    if not args.stats_only:
        seed_runs: list[dict[str, object]] = []
        base_seed = 20260504
        for run_idx in range(max(1, args.seeds)):
            seed = base_seed + run_idx
            seed_runs.append(
                train_and_eval(
                    windows,
                    labels,
                    stats,
                    families,
                    p=args.p,
                    r=args.r,
                    d_model=args.d_model,
                    n_heads=args.n_heads,
                    n_layers=args.n_layers,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device_str=args.device,
                    quantize_int8=args.quantize_int8,
                    seed=seed,
                )
            )
        if len(seed_runs) == 1:
            results = seed_runs[0]
        else:
            aurocs = [float(run["padic_transformer"]["auroc"]) for run in seed_runs]
            results = {
                "runs": seed_runs,
                "multi_seed": {
                    "seeds": [base_seed + i for i in range(len(seed_runs))],
                    "auroc_mean": float(sum(aurocs) / len(aurocs)),
                    "auroc_std": float((sum((x - (sum(aurocs) / len(aurocs))) ** 2 for x in aurocs) / len(aurocs)) ** 0.5),
                },
                "padic_transformer": seed_runs[-1]["padic_transformer"],
            }
            if "isolation_forest" in seed_runs[-1]:
                results["isolation_forest"] = seed_runs[-1]["isolation_forest"]
            if "attack_family_summary" in seed_runs[-1]:
                results["attack_family_summary"] = seed_runs[-1]["attack_family_summary"]
            if "int8_cpu_benchmark" in seed_runs[-1]:
                results["int8_cpu_benchmark"] = seed_runs[-1]["int8_cpu_benchmark"]
            if "streaming_inference" in seed_runs[-1]:
                results["streaming_inference"] = seed_runs[-1]["streaming_inference"]

    report = {
        "dataset": args.dataset,
        "stats": stats,
        "ultrametric": um_results,
        "results": results,
    }
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(out_md, args.dataset, stats, um_results, results)
    print(f"Wrote {out_json.relative_to(REPO)}")
    print(f"Wrote {out_md.relative_to(REPO)}")


if __name__ == "__main__":
    main()

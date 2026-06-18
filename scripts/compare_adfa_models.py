#!/usr/bin/env python3
"""Compare vanilla Hensel transformer and PDAIC attention on ADFA-LD."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parent if (HERE.parent / "src").exists() else HERE
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from padic_transformer.losses import AnomalyLoss
from padic_transformer.metrics import binary_auroc
from padic_transformer.model import PadicAnomalyDetector
from padic_transformer.padic_attention import PadicAttentionAnomalyDetector
from scripts.run_open_dataset import load_adfa_ld, resolve_device, safe_results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data/adfa")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-digit", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output-json", default="results/cpu_adfa_comp.json")
    parser.add_argument("--output-md", default="results/cpu_adfa_comp.md")
    return parser.parse_args()


def _f1_at_zero(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits >= 0).long()
    labs = labels.long()
    tp = int(((preds == 1) & (labs == 1)).sum())
    fp = int(((preds == 1) & (labs == 0)).sum())
    fn = int(((preds == 0) & (labs == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2.0 * precision * recall / max(1e-9, precision + recall)


def _take_subset(
    windows: torch.Tensor,
    labels: torch.Tensor,
    limit: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if limit <= 0 or limit >= labels.numel():
        return windows, labels
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(labels.numel(), generator=generator)[:limit]
    return windows[indices], labels[indices]


def _evaluate(
    name: str,
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    all_logits = []
    all_labels = []
    metric_sums: dict[str, float] = {}
    metric_count = 0
    with torch.no_grad():
        for windows_batch, labels_batch in val_loader:
            windows_batch = windows_batch.to(device)
            if hasattr(model, "forward_with_attention"):
                logits, _, _, attn_metrics = model.forward_with_attention(
                    windows_batch,
                    return_metrics=True,
                    return_features=True,
                )
                for key, value in attn_metrics.items():
                    if torch.isfinite(value):
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value.cpu().item())
                metric_count += 1
            else:
                logits, _ = model.forward_with_features(windows_batch)
            all_logits.append(logits.cpu())
            all_labels.append(labels_batch.cpu())

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    auroc = binary_auroc(logits_cat, labels_cat.long())
    f1 = _f1_at_zero(logits_cat, labels_cat)
    metrics = {
        key: value / max(1, metric_count)
        for key, value in metric_sums.items()
    }
    metrics["model"] = name
    return auroc, f1, metrics


def _train_one(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    p: int,
    alpha: float,
    pos_weight: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> dict[str, object]:
    model = model.to(device)
    loss_fn = AnomalyLoss(p=p, alpha=alpha, pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_epoch: dict[str, object] = {}
    history: list[dict[str, object]] = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for windows_batch, labels_batch in train_loader:
            windows_batch = windows_batch.to(device)
            labels_batch = labels_batch.to(device)
            logits, reps = model.forward_with_features(windows_batch)
            loss, _, _ = loss_fn(logits, labels_batch, reps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1

        auroc, f1, attn_metrics = _evaluate(name, model, val_loader, device)
        record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, batches),
            "auroc": auroc,
            "f1": f1,
            **attn_metrics,
        }
        history.append(record)
        if not best_epoch or auroc > float(best_epoch["auroc"]):
            best_epoch = record

        suffix = ""
        if "hierarchy_gap" in record:
            suffix = (
                f" hgap={float(record['hierarchy_gap']):.4f}"
                f" tp_corr={float(record.get('twin_prime_stress_padic_attention_corr', 0.0)):.4f}"
                f" gate={float(record.get('padic_gate', 0.0)):.4f}"
            )
        print(
            f"{name} epoch={epoch} train_loss={float(record['train_loss']):.4f}"
            f" auroc={auroc:.4f} f1={f1:.4f}{suffix}",
            flush=True,
        )

    elapsed = time.perf_counter() - start
    return {
        "best": best_epoch,
        "history": history,
        "elapsed_s": elapsed,
    }


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    dataset = report["dataset"]
    vanilla = report["results"]["vanilla"]["best"]
    pdaic = report["results"]["pdaic_attention"]["best"]
    rows = [
        "# CPU ADFA Comparison",
        "",
        f"- Windows: `{dataset['windows']}`",
        f"- Train windows: `{dataset['train_windows']}`",
        f"- Val windows: `{dataset['val_windows']}`",
        f"- Attack windows: `{dataset['attack_windows']}`",
        f"- pos_weight: `{dataset['pos_weight']:.4f}`",
        f"- Epochs: `{report['config']['epochs']}`",
        f"- Batch size: `{report['config']['batch_size']}`",
        f"- Loss alpha: `{report['config']['alpha']}`",
        "",
        "| Model | Best epoch | Best AUROC | Best F1 | hierarchy gap | twin-prime corr | p-adic gate | elapsed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, best, elapsed in [
        ("vanilla", vanilla, report["results"]["vanilla"]["elapsed_s"]),
        ("pdaic_attention", pdaic, report["results"]["pdaic_attention"]["elapsed_s"]),
    ]:
        rows.append(
            "| {label} | {epoch} | {auroc:.4f} | {f1:.4f} | {hgap:.4f} | {tp:.4f} | {gate:.4f} | {elapsed:.1f}s |".format(
                label=label,
                epoch=int(best["epoch"]),
                auroc=float(best["auroc"]),
                f1=float(best["f1"]),
                hgap=float(best.get("hierarchy_gap", 0.0)),
                tp=float(best.get("twin_prime_stress_padic_attention_corr", 0.0)),
                gate=float(best.get("padic_gate", 0.0)),
                elapsed=float(elapsed),
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    windows, labels, stats, _families = load_adfa_ld(
        Path(args.data_dir),
        p=args.p,
        r=args.r,
        window_size=args.window_size,
        stride=args.stride,
        download=not args.no_download,
    )
    n = labels.numel()
    n_train = int(0.8 * n)
    n_val = n - n_train
    split = random_split(
        TensorDataset(windows, labels),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_ds, val_ds = split
    train_windows, train_labels = windows[train_ds.indices], labels[train_ds.indices]
    val_windows, val_labels = windows[val_ds.indices], labels[val_ds.indices]
    train_windows, train_labels = _take_subset(
        train_windows,
        train_labels,
        args.max_train_windows,
        args.seed + 1,
    )
    val_windows, val_labels = _take_subset(
        val_windows,
        val_labels,
        args.max_val_windows,
        args.seed + 2,
    )

    print(
        f"ADFA windows={n} train={train_labels.numel()} val={val_labels.numel()}"
        f" attacks={int(labels.sum().item())} pos_weight={float(stats['pos_weight']):.4f}"
        f" device={device}",
        flush=True,
    )

    train_loader = DataLoader(
        TensorDataset(train_windows, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_windows, val_labels),
        batch_size=args.batch_size * 2,
        shuffle=False,
    )

    common = {
        "p": args.p,
        "r": args.r,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "ffn_dim": args.d_model * 4,
        "head_hidden": args.d_model // 2,
        "max_seq_len": args.window_size,
    }
    vanilla = PadicAnomalyDetector(**common)
    pdaic = PadicAttentionAnomalyDetector(
        **common,
        d_digit=args.d_digit,
    )

    results = {
        "vanilla": _train_one(
            "vanilla",
            vanilla,
            train_loader,
            val_loader,
            p=args.p,
            alpha=args.alpha,
            pos_weight=float(stats["pos_weight"]),
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        ),
        "pdaic_attention": _train_one(
            "pdaic_attention",
            pdaic,
            train_loader,
            val_loader,
            p=args.p,
            alpha=args.alpha,
            pos_weight=float(stats["pos_weight"]),
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        ),
    }
    report: dict[str, object] = {
        "dataset": {
            "windows": int(n),
            "train_windows": int(train_labels.numel()),
            "val_windows": int(val_labels.numel()),
            "attack_windows": int(labels.sum().item()),
            "pos_weight": float(stats["pos_weight"]),
            "p": args.p,
            "r": args.r,
            "window_size": args.window_size,
            "stride": args.stride,
        },
        "config": vars(args),
        "results": results,
    }
    json_path = safe_results_path(args.output_json)
    md_path = safe_results_path(args.output_md)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    print(f"Wrote {json_path.relative_to(REPO)}")
    print(f"Wrote {md_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()

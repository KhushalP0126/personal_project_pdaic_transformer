"""Training loop for the p-adic anomaly detector."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import binary_auroc
from .model_fixes import compute_diversity_regularization, log_temperature_health

REPO_ROOT = Path(__file__).resolve().parents[2]


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    major = torch.cuda.get_device_capability(device)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def _search_threshold_metrics(
    all_logits: torch.Tensor,
    all_labels: torch.Tensor,
    num_thresholds: int = 200,
) -> dict[str, float]:
    labs = all_labels.long()

    thresholds = torch.linspace(all_logits.min().item(), all_logits.max().item(), steps=num_thresholds)
    best = {
        "threshold": 0.0,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    neg_total = max(1, int((labs == 0).sum()))
    pos_total = max(1, int((labs == 1).sum()))

    for threshold in thresholds:
        preds = (all_logits >= threshold).long()
        tp = int(((preds == 1) & (labs == 1)).sum())
        fp = int(((preds == 1) & (labs == 0)).sum())
        fn = int(((preds == 0) & (labs == 1)).sum())
        tn = int(((preds == 0) & (labs == 0)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        fpr = fp / neg_total
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold.item()),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

    best["neg_total"] = neg_total
    best["pos_total"] = pos_total
    return best


def _compute_metrics(all_logits: torch.Tensor, all_labels: torch.Tensor, threshold: float = 0.0) -> dict[str, float]:
    preds = (all_logits >= threshold).long()
    labs = all_labels.long()

    tp = int(((preds == 1) & (labs == 1)).sum())
    fp = int(((preds == 1) & (labs == 0)).sum())
    fn = int(((preds == 0) & (labs == 1)).sum())
    tn = int(((preds == 0) & (labs == 0)).sum())

    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    auroc = binary_auroc(all_logits, labs)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _dataset_pos_weight(ds) -> float:
    labels = getattr(ds, "labels", None)
    if labels is None:
        return 1.0
    n_pos = int(labels.sum().item())
    n_total = int(labels.numel())
    n_neg = n_total - n_pos
    return n_neg / max(1, n_pos)


@dataclass
class TrainConfig:
    p: int = 3
    r: int = 16
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ffn_dim: int = 1024
    head_hidden: int = 128
    d_digit: int = 16
    dropout: float = 0.1

    window_size: int = 32
    attack_fraction: float = 0.3
    attack_min_len: int = 2
    attack_max_len: int = 8
    hierarchy_rule_dataset: bool = False
    rule_subtree_depth: int = 2
    rule_stay_steps: int = 4
    rule_attack_tokens: int = 1
    realistic_dataset: bool = False
    realistic_attack_fraction: float = 0.005
    idle_fraction: float = 0.70
    attack_kinds: tuple[str, ...] = ("cross_class", "stuck_at", "burst", "ordering")
    n_train: int = 65536
    n_val: int = 8192
    samples: int = 16384
    classes: int = 32
    tokens_per_class: int = 128
    seed: int = 20260504
    max_seq_len: int = 256

    epochs: int = 20
    batch_size: int = 512
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    warmup_epochs: int = 2
    num_workers: int = 4

    alpha: float = 0.5
    pos_weight: float = 1.0
    margin_pos: float = 0.1
    margin_neg: float = 0.5
    max_pairs: int = 4096

    checkpoint_dir: str = "results/checkpoints"
    log_json: str | None = "results/training_log.json"
    log_md: str = "results/training_log.md"
    save_every: int = 5


def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = config.warmup_epochs
    total_steps = config.epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_steps:
            return float(epoch + 1) / float(max(1, warmup_steps))
        progress = (epoch - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_grad_norm: float,
    grad_accum: int,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    if grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")
    model.train()
    total_loss = total_bce = total_ctr = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for step, (digits, labels) in enumerate(loader):
        digits = digits.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            logits, representations = model.forward_with_features(digits)
            loss, bce_loss, ctr_loss = loss_fn(logits, labels, representations)
            loss = loss + compute_diversity_regularization(model)
            loss = loss / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_last = step + 1 == len(loader)
        if (step + 1) % grad_accum == 0 or is_last:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += float(loss.item()) * grad_accum
        total_bce += float(bce_loss.item())
        total_ctr += float(ctr_loss.item())
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "bce": total_bce / max(1, n_batches),
        "contrastive": total_ctr / max(1, n_batches),
    }


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    model.eval()
    total_loss = total_bce = total_ctr = 0.0
    n_batches = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    attn_metric_sums: dict[str, float] = {}
    attn_metric_count = 0

    for digits, labels in loader:
        digits = digits.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attn_metrics: dict[str, torch.Tensor] = {}
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(amp_dtype != torch.float32),
        ):
            # Validation can request attention metrics from the same forward pass.
            if hasattr(model, "forward_with_attention"):
                logits, representations, _, attn_metrics = model.forward_with_attention(
                    digits,
                    return_metrics=True,
                    return_features=True,
                )
                loss, bce_loss, ctr_loss = loss_fn(logits, labels, representations)
            else:
                logits, representations = model.forward_with_features(digits)
                loss, bce_loss, ctr_loss = loss_fn(logits, labels, representations)
        total_loss += float(loss.item())
        total_bce += float(bce_loss.item())
        total_ctr += float(ctr_loss.item())
        n_batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        if attn_metrics:
            for key, value in attn_metrics.items():
                attn_metric_sums[key] = attn_metric_sums.get(key, 0.0) + float(value.detach().cpu().item())
            attn_metric_count += 1

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    metrics = _compute_metrics(logits_cat, labels_cat)
    normal_scores = logits_cat[labels_cat.long() == 0]
    anom_scores = logits_cat[labels_cat.long() == 1]
    score_stats = {
        "normal_mean": float(normal_scores.mean().item()) if normal_scores.numel() else float("nan"),
        "normal_std": float(normal_scores.std().item()) if normal_scores.numel() > 1 else 0.0,
        "anom_mean": float(anom_scores.mean().item()) if anom_scores.numel() else float("nan"),
        "anom_std": float(anom_scores.std().item()) if anom_scores.numel() > 1 else 0.0,
        "score_gap": float(anom_scores.mean().item() - normal_scores.mean().item())
        if normal_scores.numel() and anom_scores.numel()
        else float("nan"),
    }
    threshold_metrics = _search_threshold_metrics(logits_cat, labels_cat)
    metrics["threshold_search"] = threshold_metrics
    metrics["best_f1"] = threshold_metrics["f1"]
    metrics["best_precision"] = threshold_metrics["precision"]
    metrics["best_recall"] = threshold_metrics["recall"]
    metrics["best_fpr"] = threshold_metrics["fpr"]
    metrics["best_threshold"] = threshold_metrics["threshold"]
    metrics.update(score_stats)
    if attn_metric_count > 0:
        for key, value in attn_metric_sums.items():
            metrics[key] = value / attn_metric_count
    metrics["loss"] = total_loss / max(1, n_batches)
    metrics["bce"] = total_bce / max(1, n_batches)
    metrics["contrastive"] = total_ctr / max(1, n_batches)
    return metrics


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": config.__dict__,
        },
        path,
    )


def _safe_results_path(raw: str) -> Path:
    path = (REPO_ROOT / raw).resolve()
    results_root = REPO_ROOT / "results"
    if results_root not in (path, *path.parents):
        raise ValueError("outputs must be under results/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _make_optimizer(
    model: nn.Module,
    device: torch.device,
    optimizer_kwargs: dict[str, float],
) -> torch.optim.Optimizer:
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(
                model.parameters(),
                fused=True,
                **optimizer_kwargs,
            )
        except (TypeError, RuntimeError):
            return torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    return torch.optim.AdamW(model.parameters(), **optimizer_kwargs)


def train(
    config: TrainConfig,
    device: torch.device,
    model_factory: Callable[[TrainConfig], nn.Module] | None = None,
) -> dict[str, object]:
    from .config import BenchmarkConfig
    from .dataset import AnomalyDatasetConfig, build_dataloaders
    from .dataset_hierarchy_rules import HierarchyRuleDataset, HierarchyRuleDatasetConfig
    from .dataset_realistic import RealisticBusDataset, RealisticDatasetConfig, make_weighted_loss
    from .losses import AnomalyLoss
    from .model import PadicAnomalyDetector
    from .ultrametric import generate_clustered_hensel_dataset
    from torch.utils.data import DataLoader

    print(f"\n{'='*60}")
    print("P-ADIC ANOMALY DETECTOR -- TRAINING")
    print(f"  device : {device}")
    print(f"  p={config.p}  r={config.r}  d_model={config.d_model}")
    print(f"  epochs={config.epochs}  batch={config.batch_size}")
    print(f"{'='*60}\n")

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    benchmark_cfg = BenchmarkConfig(
        p=config.p,
        r=config.r,
        samples=config.samples,
        classes=config.classes,
        tokens_per_class=config.tokens_per_class,
        seed=config.seed,
    )
    print("Building dataloaders...")
    if config.hierarchy_rule_dataset:
        train_hensel = generate_clustered_hensel_dataset(benchmark_cfg, device="cpu")
        val_cfg = BenchmarkConfig(
            p=config.p,
            r=config.r,
            samples=config.samples,
            classes=config.classes,
            tokens_per_class=config.tokens_per_class,
            seed=config.seed + 999_999,
            triplets=benchmark_cfg.triplets,
            distance_pairs=benchmark_cfg.distance_pairs,
        )
        val_hensel = generate_clustered_hensel_dataset(val_cfg, device="cpu")
        rule_cfg = HierarchyRuleDatasetConfig(
            window_size=config.window_size,
            attack_fraction=config.attack_fraction,
            subtree_depth=config.rule_subtree_depth,
            stay_steps=config.rule_stay_steps,
            attack_tokens=config.rule_attack_tokens,
            seed=config.seed,
        )
        train_ds = HierarchyRuleDataset(train_hensel, rule_cfg, n_samples=config.n_train)
        val_ds = HierarchyRuleDataset(
            val_hensel,
            HierarchyRuleDatasetConfig(
                window_size=config.window_size,
                attack_fraction=config.attack_fraction,
                subtree_depth=config.rule_subtree_depth,
                stay_steps=config.rule_stay_steps,
                attack_tokens=config.rule_attack_tokens,
                seed=config.seed ^ 0xA11CE,
            ),
            n_samples=config.n_val,
        )
        pin = device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin,
            persistent_workers=(config.num_workers > 0),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin,
            persistent_workers=(config.num_workers > 0),
            drop_last=False,
        )
    elif config.realistic_dataset:
        train_hensel = generate_clustered_hensel_dataset(benchmark_cfg, device="cpu")
        val_cfg = BenchmarkConfig(
            p=config.p,
            r=config.r,
            samples=config.samples,
            classes=config.classes,
            tokens_per_class=config.tokens_per_class,
            seed=config.seed + 999_999,
            triplets=benchmark_cfg.triplets,
            distance_pairs=benchmark_cfg.distance_pairs,
        )
        val_hensel = generate_clustered_hensel_dataset(val_cfg, device="cpu")
        realistic_cfg = RealisticDatasetConfig(
            window_size=config.window_size,
            attack_fraction=config.realistic_attack_fraction,
            idle_fraction=config.idle_fraction,
            attack_min_len=config.attack_min_len,
            attack_max_len=config.attack_max_len,
            attack_kinds=config.attack_kinds,
            seed=config.seed,
        )
        train_ds = RealisticBusDataset(train_hensel, realistic_cfg, n_samples=config.n_train)
        val_ds = RealisticBusDataset(val_hensel, realistic_cfg, n_samples=config.n_val)
        pin = device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin,
            persistent_workers=(config.num_workers > 0),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin,
            persistent_workers=(config.num_workers > 0),
            drop_last=False,
        )
    else:
        anomaly_cfg = AnomalyDatasetConfig(
            window_size=config.window_size,
            attack_fraction=config.attack_fraction,
            attack_min_len=config.attack_min_len,
            attack_max_len=config.attack_max_len,
            seed=config.seed,
        )
        train_loader, val_loader = build_dataloaders(
            benchmark_cfg=benchmark_cfg,
            anomaly_cfg=anomaly_cfg,
            n_train=config.n_train,
            n_val=config.n_val,
            batch_size=config.batch_size,
            device=device,
            num_workers=config.num_workers,
        )

    if model_factory is None:
        model = PadicAnomalyDetector(
            p=config.p,
            r=config.r,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            ffn_dim=config.ffn_dim,
            head_hidden=config.head_hidden,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )
    else:
        model = model_factory(config)
    model = model.to(device)
    print(model.parameter_summary())

    if config.realistic_dataset:
        loss_fn = make_weighted_loss(
            train_ds,
            p=config.p,
            alpha=config.alpha,
            margin_pos=config.margin_pos,
            margin_neg=config.margin_neg,
            max_pairs=config.max_pairs,
        ).to(device)
    else:
        pos_weight = _dataset_pos_weight(train_loader.dataset)
        print(f"synthetic pos_weight={pos_weight:.2f} (n_neg/n_pos)")
        loss_fn = AnomalyLoss(
            p=config.p,
            alpha=config.alpha,
            pos_weight=pos_weight,
            margin_pos=config.margin_pos,
            margin_neg=config.margin_neg,
            max_pairs=config.max_pairs,
        ).to(device)

    optimizer_kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    optimizer = _make_optimizer(model, device, optimizer_kwargs)
    scheduler = _build_scheduler(optimizer, config)

    amp_dtype = _amp_dtype(device)
    print(f"\nAMP dtype: {amp_dtype}")
    scaler: torch.amp.GradScaler | None = None
    if device.type == "cuda" and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")

    ckpt_dir = _safe_results_path(config.checkpoint_dir)
    log_json_path = _safe_results_path(config.log_json) if config.log_json else None
    log_md_path = _safe_results_path(config.log_md)

    history: list[dict] = []
    best_auroc = -1.0
    best_epoch = 0
    run_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        t0 = time.perf_counter()
        lr_now = optimizer.param_groups[0]["lr"]

        train_metrics = _train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            amp_dtype,
            config.max_grad_norm,
            config.gradient_accumulation_steps,
            scaler,
        )
        val_metrics = _val_epoch(model, val_loader, loss_fn, device, amp_dtype)
        log_temperature_health(model, epoch)
        scheduler.step()
        lr_next = optimizer.param_groups[0]["lr"]

        elapsed = time.perf_counter() - t0
        attn_suffix = ""
        if "hierarchy_gap" in val_metrics and "padic_gate" in val_metrics:
            attn_suffix = (
                f"hgap={val_metrics['hierarchy_gap']:.4f}  "
                f"gate={val_metrics['padic_gate']:.4f}  "
            )

        epoch_record = {
            "epoch": epoch,
            "lr": lr_now,
            "lr_used": lr_now,
            "lr_next": lr_next,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_s": elapsed,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:>3}/{config.epochs} | "
            f"lr={lr_now:.2e} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"acc={val_metrics['accuracy']:.4f}  "
            f"best_f1(anomaly)={val_metrics['best_f1']:.4f}  "
            f"auroc={val_metrics['auroc']:.4f}  "
            f"gap={val_metrics['score_gap']:.4f}  "
            f"{attn_suffix}"
            f"[{elapsed:.1f}s]"
        )

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_epoch = epoch
            _save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, val_metrics, config)
            print(f"  New best AUROC={best_auroc:.4f} -- saved best.pt")

        if epoch % config.save_every == 0:
            _save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, val_metrics, config)

    _save_checkpoint(ckpt_dir / "final.pt", model, optimizer, config.epochs, history[-1]["val"], config)

    total_time = time.perf_counter() - run_start
    best_record = next(r for r in history if r["epoch"] == best_epoch)
    result = {
        "best_epoch": best_epoch,
        "best_auroc": best_auroc,
        "best_f1": best_record["val"]["best_f1"],
        "best_precision": best_record["val"]["best_precision"],
        "best_recall": best_record["val"]["best_recall"],
        "best_fpr": best_record["val"]["best_fpr"],
        "total_seconds": total_time,
        "history": history,
    }

    if log_json_path is not None:
        log_json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_training_log(log_md_path, config, result, device)

    print(f"\nTraining complete in {total_time:.1f}s -- best AUROC {best_auroc:.4f} at epoch {best_epoch}")
    if log_json_path is not None:
        print(f"Logs: {log_json_path.relative_to(REPO_ROOT)}")
        print(f"      {log_md_path.relative_to(REPO_ROOT)}")
    else:
        print(f"Logs: {log_md_path.relative_to(REPO_ROOT)}")
    return result


def _write_training_log(path: Path, config: TrainConfig, result: dict, device: torch.device) -> None:
    history = result["history"]
    header = "\n".join(
        [
            "# P-Adic Anomaly Detector -- Training Log",
            "",
            f"- **Device**: `{device}`",
            f"- **p**: `{config.p}`  **r**: `{config.r}`  **d_model**: `{config.d_model}`",
            f"- **Epochs**: `{config.epochs}`  **Batch**: `{config.batch_size}`",
            f"- **Best AUROC**: `{result['best_auroc']:.4f}` at epoch `{result['best_epoch']}`",
            f"- **Best F1 (anomaly)**: `{result['best_f1']:.4f}`",
            f"- **Best Precision**: `{result['best_precision']:.4f}`",
            f"- **Best Recall**: `{result['best_recall']:.4f}`",
            f"- **Best FPR**: `{result['best_fpr']:.4f}`",
            f"- **Total time**: `{result['total_seconds']:.1f}s`",
            "",
            "| Epoch | LR | Train Loss | Val Loss | Accuracy | Best F1 (anomaly) | Precision | Recall | FPR | AUROC |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    rows = [
        f"| {r['epoch']} | {r['lr']:.2e} | {r['train']['loss']:.4f} | "
        f"{r['val']['loss']:.4f} | {r['val']['accuracy']:.4f} | "
        f"{r['val']['best_f1']:.4f} | {r['val']['best_precision']:.4f} | "
        f"{r['val']['best_recall']:.4f} | {r['val']['best_fpr']:.4f} | {r['val']['auroc']:.4f} |"
        for r in history
    ]
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

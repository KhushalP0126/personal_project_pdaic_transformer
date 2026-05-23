"""Baselines and real-data loading helpers for p-adic anomaly detection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .hensel import int64_to_digits
from .metrics import binary_auroc


def _flatten_windows(windows: torch.Tensor) -> torch.Tensor:
    return windows.reshape(windows.shape[0], -1).to(torch.float32)


def _scores_to_f1(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> float:
    preds = (scores >= threshold).to(torch.int64)
    labs = labels.to(torch.int64)
    tp = int(((preds == 1) & (labs == 1)).sum().item())
    fp = int(((preds == 1) & (labs == 0)).sum().item())
    fn = int(((preds == 0) & (labs == 1)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2.0 * precision * recall / max(1e-9, precision + recall)


def _token_ids_from_digits(windows: torch.Tensor, p: int) -> torch.Tensor:
    r = windows.shape[-1]
    powers = torch.tensor([p**i for i in range(r)], dtype=torch.int64, device=windows.device)
    return (windows * powers).sum(dim=-1)


def _remap_hierarchy_windows(
    windows: torch.Tensor,
    p: int,
    variant: str,
    seed: int,
) -> torch.Tensor:
    if variant == "true":
        return windows
    if variant not in {"shuffled", "random"}:
        raise ValueError(f"unknown hierarchy variant: {variant}")

    ids = _token_ids_from_digits(windows, p)
    flat_ids = ids.reshape(-1)
    unique_ids = torch.unique(flat_ids, sorted=True)
    rng = torch.Generator(device=windows.device)
    rng.manual_seed(seed)

    if variant == "shuffled":
        remapped_vocab = unique_ids[torch.randperm(unique_ids.numel(), generator=rng, device=windows.device)]
    else:
        random_digits = torch.randint(
            0,
            p,
            (unique_ids.numel(), windows.shape[-1]),
            dtype=torch.int64,
            device=windows.device,
            generator=rng,
        )
        remapped_vocab = _token_ids_from_digits(random_digits, p).reshape(unique_ids.numel())

    remap_indices = torch.searchsorted(unique_ids, flat_ids)
    remapped_ids = remapped_vocab[remap_indices]
    remapped_digits = int64_to_digits(remapped_ids, p=p, r=windows.shape[-1]).reshape_as(windows)
    return remapped_digits


def remap_hierarchy_windows(
    windows: torch.Tensor,
    p: int,
    variant: str,
    seed: int,
) -> torch.Tensor:
    return _remap_hierarchy_windows(windows, p, variant, seed)


def run_majority_baseline(train_labels: torch.Tensor, val_labels: torch.Tensor) -> dict[str, float]:
    train_pos = float(train_labels.mean().item())
    majority = 1 if train_pos >= 0.5 else 0
    preds = torch.full_like(val_labels, float(majority))
    accuracy = float((preds == val_labels).to(torch.float32).mean().item())
    f1 = _scores_to_f1(preds, val_labels, threshold=0.5 if majority == 1 else 1.0)
    return {"auroc": 0.5, "f1": f1, "accuracy": accuracy, "predicted_class": float(majority)}


class FlattenClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_flatten_baseline(
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    val_windows: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight: float,
    device: torch.device,
) -> dict[str, float]:
    train_x = _flatten_windows(train_windows)
    val_x = _flatten_windows(val_windows)
    model = FlattenClassifier(train_x.shape[1], hidden_dim=hidden_dim).to(device)
    train_ds = TensorDataset(train_x, train_labels)
    val_ds = TensorDataset(val_x, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best = {"auroc": 0.5, "f1": 0.0}
    t0 = time.perf_counter()

    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                logits = model(x_batch.to(device))
                all_logits.append(logits.cpu())
                all_labels.append(y_batch.cpu())
        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        auroc = binary_auroc(logits_cat, labels_cat)
        f1 = _scores_to_f1(logits_cat, labels_cat, threshold=0.0)
        if auroc > best["auroc"]:
            best = {"auroc": auroc, "f1": f1}

    best["train_time_s"] = time.perf_counter() - t0
    return best


def run_logistic_regression_baseline(
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    val_windows: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    pos_weight: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _train_flatten_baseline(
        train_windows,
        train_labels,
        val_windows,
        val_labels,
        hidden_dim=0,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        pos_weight=pos_weight,
        device=device,
    )


def run_mlp_baseline(
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    val_windows: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    hidden_dim: int = 256,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    pos_weight: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _train_flatten_baseline(
        train_windows,
        train_labels,
        val_windows,
        val_labels,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        pos_weight=pos_weight,
        device=device,
    )


def run_isolation_forest_baseline(
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    val_windows: torch.Tensor,
    val_labels: torch.Tensor,
) -> dict[str, float]:
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import f1_score, roc_auc_score
        import numpy as np
    except ImportError:
        print("sklearn not installed. Run: pip install scikit-learn")
        return {}

    X_train = train_windows.reshape(train_windows.shape[0], -1).numpy().astype(np.float32)
    X_val = val_windows.reshape(val_windows.shape[0], -1).numpy().astype(np.float32)
    y_val = val_labels.numpy().astype(int)
    normal_mask = train_labels.numpy() == 0
    X_train_normal = X_train[normal_mask]

    t0 = time.perf_counter()
    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    clf.fit(X_train_normal)
    train_time = time.perf_counter() - t0

    scores = -clf.score_samples(X_val)
    preds = (clf.predict(X_val) == -1).astype(int)
    auroc = float(roc_auc_score(y_val, scores)) if len(set(y_val)) > 1 else 0.5
    f1 = float(f1_score(y_val, preds, zero_division=0))
    return {"auroc": auroc, "f1": f1, "train_time_s": train_time}


class StandardTransformerDetector(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        head_hidden: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    @staticmethod
    def digits_to_ids(windows: torch.Tensor, p: int) -> torch.Tensor:
        r = windows.shape[-1]
        powers = torch.tensor([p**i for i in range(r)], dtype=torch.int64, device=windows.device)
        return (windows * powers).sum(dim=-1)

    def forward_with_features(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embed(token_ids) + self.pos_embed(positions).unsqueeze(0)
        x = self.embed_drop(x)
        h = self.encoder(x, src_key_padding_mask=padding_mask)
        pooled = h.mean(dim=1)
        logits = self.head(pooled).squeeze(-1)
        return logits, pooled

    def forward(self, token_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self.forward_with_features(token_ids, padding_mask)
        return logits

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


def run_standard_transformer_baseline(
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    val_windows: torch.Tensor,
    val_labels: torch.Tensor,
    p: int,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    pos_weight: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    r = train_windows.shape[-1]
    vocab_size = p**r
    model = StandardTransformerDetector(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=d_model * 4,
        head_hidden=d_model // 2,
    ).to(device)

    train_ids = StandardTransformerDetector.digits_to_ids(train_windows, p)
    val_ids = StandardTransformerDetector.digits_to_ids(val_windows, p)

    train_ds = TensorDataset(train_ids, train_labels)
    val_ds = TensorDataset(val_ids, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best_auroc = 0.0
    t0 = time.perf_counter()

    for _ in range(epochs):
        model.train()
        for ids_batch, labels_batch in train_loader:
            ids_batch = ids_batch.to(device)
            labels_batch = labels_batch.to(device)
            logits, _ = model.forward_with_features(ids_batch)
            loss = criterion(logits, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for ids_batch, labels_batch in val_loader:
                logits, _ = model.forward_with_features(ids_batch.to(device))
                all_logits.append(logits.cpu())
                all_labels.append(labels_batch.cpu())
        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        auroc = binary_auroc(logits_cat, labels_cat)
        best_auroc = max(best_auroc, auroc)

    elapsed = time.perf_counter() - t0
    return {"auroc": best_auroc, "train_time_s": elapsed}


def _eval_digit_window_batch(
    model: nn.Module,
    windows_batch: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if hasattr(model, "forward_with_attention"):
        logits, _, _, attn_metrics = model.forward_with_attention(
            windows_batch,
            return_metrics=True,
            return_features=True,
        )
        return logits, attn_metrics
    logits, _ = model.forward_with_features(windows_batch)
    return logits, {}


def _train_digit_window_model(
    model: nn.Module,
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    val_windows: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight: float,
    device: torch.device,
) -> dict[str, float]:
    train_ds = TensorDataset(train_windows, train_labels)
    val_ds = TensorDataset(val_windows, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best: dict[str, float] = {"auroc": 0.5, "f1": 0.0}
    t0 = time.perf_counter()

    for _ in range(epochs):
        model.train()
        for windows_batch, labels_batch in train_loader:
            windows_batch = windows_batch.to(device)
            labels_batch = labels_batch.to(device)
            logits, _ = model.forward_with_features(windows_batch)
            loss = criterion(logits, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_logits, all_labels = [], []
        metric_sums: dict[str, float] = {}
        metric_count = 0
        with torch.no_grad():
            for windows_batch, labels_batch in val_loader:
                windows_batch = windows_batch.to(device)
                logits, attn_metrics = _eval_digit_window_batch(model, windows_batch)
                all_logits.append(logits.cpu())
                all_labels.append(labels_batch.cpu())
                if attn_metrics:
                    for key, value in attn_metrics.items():
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach().cpu().item())
                    metric_count += 1
        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        auroc = binary_auroc(logits_cat, labels_cat)
        f1 = _scores_to_f1(logits_cat, labels_cat, threshold=0.0)
        if auroc > best["auroc"]:
            best = {"auroc": auroc, "f1": f1}
            if metric_count > 0:
                for key, value in metric_sums.items():
                    best[key] = value / metric_count

    best["train_time_s"] = time.perf_counter() - t0
    return best


def run_hensel_transformer_baseline(
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    val_windows: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    p: int,
    r: int,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    pos_weight: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from .model import PadicAnomalyDetector

    model = PadicAnomalyDetector(
        p=p,
        r=r,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=d_model * 4,
        head_hidden=d_model // 2,
    )
    return _train_digit_window_model(
        model.to(device),
        train_windows,
        train_labels,
        val_windows,
        val_labels,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        pos_weight=pos_weight,
        device=device,
    )


def run_padic_attention_baseline(
    train_windows: torch.Tensor,
    train_labels: torch.Tensor,
    val_windows: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    p: int,
    r: int,
    hierarchy_variant: str = "true",
    seed: int = 20260504,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    d_digit: int = 16,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    pos_weight: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from .padic_attention import PadicAttentionAnomalyDetector

    train_variant = _remap_hierarchy_windows(train_windows, p, hierarchy_variant, seed)
    val_variant = _remap_hierarchy_windows(val_windows, p, hierarchy_variant, seed + 1)
    model = PadicAttentionAnomalyDetector(
        p=p,
        r=r,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=d_model * 4,
        head_hidden=d_model // 2,
        d_digit=d_digit,
    ).to(device)
    result = _train_digit_window_model(
        model,
        train_variant,
        train_labels,
        val_variant,
        val_labels,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        pos_weight=pos_weight,
        device=device,
    )
    result["hierarchy_variant"] = hierarchy_variant
    return result


def evaluate_attention_model(
    model: nn.Module,
    windows: torch.Tensor,
    labels: torch.Tensor,
    *,
    p: int,
    hierarchy_variant: str = "true",
    seed: int = 20260504,
    batch_size: int = 256,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    eval_windows = _remap_hierarchy_windows(windows, p, hierarchy_variant, seed)
    loader = DataLoader(TensorDataset(eval_windows, labels), batch_size=batch_size, shuffle=False)
    all_logits, all_labels = [], []
    metric_sums: dict[str, float] = {}
    metric_count = 0
    with torch.no_grad():
        for windows_batch, labels_batch in loader:
            windows_batch = windows_batch.to(device)
            logits, attn_metrics = _eval_digit_window_batch(model, windows_batch)
            all_logits.append(logits.cpu())
            all_labels.append(labels_batch.cpu())
            if attn_metrics:
                for key, value in attn_metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach().cpu().item())
                metric_count += 1
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    result = {
        "auroc": binary_auroc(logits_cat, labels_cat),
        "f1": _scores_to_f1(logits_cat, labels_cat, threshold=0.0),
        "hierarchy_variant": hierarchy_variant,
    }
    if metric_count > 0:
        for key, value in metric_sums.items():
            result[key] = value / metric_count
    return result


DATASET_GUIDE: dict[str, dict[str, str]] = {
    "ADFA-LD": {
        "what": "Linux syscall traces from real attack scenarios.",
        "why": "Best structural match for p-adic syscall sequences.",
        "how_to_map": "Map syscall IDs to Hensel digits using your prime base mapping.",
        "url": "https://research.unsw.edu.au/projects/adfa-ids-datasets",
        "format": "Text files, one syscall ID per line",
        "size": "~130MB",
        "caveat": "Labels are trace-level, so use weak supervision.",
    },
    "BETH": {
        "what": "Large-scale Linux audit logs from honeypots.",
        "why": "Modern, larger, and rich in event types.",
        "how_to_map": "Extract event_id and encode discretely.",
        "url": "https://www.kaggle.com/datasets/katehighnam/beth-dataset",
        "format": "CSV",
        "size": "~1GB",
        "caveat": "Honeypot data can be noisy.",
    },
    "LANL_CERT": {
        "what": "Process and network logs with red-team events.",
        "why": "Closest public analog to hardware telemetry.",
        "how_to_map": "Label windows by event timestamps.",
        "url": "https://csr.lanl.gov/data/cyber1/",
        "format": "Compressed CSV",
        "size": "~12GB compressed",
        "caveat": "Requires registration and is large.",
    },
}


def print_dataset_guide() -> None:
    for name, info in DATASET_GUIDE.items():
        print(f"{name}: {info['what']} -> {info['url']}")


def load_adfa_ld(
    data_dir: str,
    syscall_map: dict[str, int] | None = None,
    p: int = 3,
    r: int = 8,
    window_size: int = 32,
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"ADFA-LD data directory not found: {data_dir}")

    def read_trace(filepath: Path) -> list[int]:
        with open(filepath, "r", encoding="utf-8") as f:
            return [int(x) for x in f.read().split() if x.strip().isdigit()]

    def trace_to_windows(syscall_ids: list[int], label: float) -> tuple[list[torch.Tensor], list[float]]:
        if len(syscall_ids) < window_size:
            return [], []
        ids_tensor = torch.tensor(syscall_ids, dtype=torch.int64)
        max_val = p**r - 1
        ids_clamped = ids_tensor.clamp(0, max_val)
        digit_seq = int64_to_digits(ids_clamped, p=p, r=r)
        wins, labs = [], []
        for start in range(0, len(syscall_ids) - window_size + 1, stride):
            wins.append(digit_seq[start : start + window_size])
            labs.append(label)
        return wins, labs

    all_windows: list[torch.Tensor] = []
    all_labels: list[float] = []

    normal_dir = data_path / "Training_Data_Master"
    if normal_dir.exists():
        for filepath in sorted(normal_dir.glob("*.txt")):
            ids = read_trace(filepath)
            wins, labs = trace_to_windows(ids, label=0.0)
            all_windows.extend(wins)
            all_labels.extend(labs)

    attack_dir = data_path / "Attack_Data_Master"
    if attack_dir.exists():
        for attack_family in sorted(attack_dir.iterdir()):
            if not attack_family.is_dir():
                continue
            for filepath in sorted(attack_family.glob("*.txt")):
                ids = read_trace(filepath)
                wins, labs = trace_to_windows(ids, label=1.0)
                all_windows.extend(wins)
                all_labels.extend(labs)

    if not all_windows:
        raise RuntimeError(f"No trace files found in {data_dir}")

    windows_tensor = torch.stack(all_windows)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
    return windows_tensor, labels_tensor

"""Baselines and real-data loading helpers for p-adic anomaly detection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .hensel import int64_to_digits


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
        labels_cat = torch.cat(all_labels).long()
        try:
            from sklearn.metrics import roc_auc_score

            auroc = float(roc_auc_score(labels_cat.numpy(), logits_cat.numpy()))
        except Exception:
            auroc = 0.5
        best_auroc = max(best_auroc, auroc)

    elapsed = time.perf_counter() - t0
    return {"auroc": best_auroc, "train_time_s": elapsed}


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

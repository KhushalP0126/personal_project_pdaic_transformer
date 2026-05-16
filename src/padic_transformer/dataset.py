"""PyTorch Dataset and DataLoader utilities for anomaly detection training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from .config import BenchmarkConfig
from .ultrametric import HenselDataset, generate_clustered_hensel_dataset


@dataclass(frozen=True)
class AnomalyDatasetConfig:
    window_size: int = 32
    attack_fraction: float = 0.3
    attack_min_len: int = 2
    attack_max_len: int = 8
    seed: int = 20260504

    def validate(self) -> None:
        if not 2 <= self.window_size <= 4096:
            raise ValueError(f"window_size must be in [2, 4096], got {self.window_size}")
        if not 0.0 < self.attack_fraction < 1.0:
            raise ValueError(f"attack_fraction must be in (0, 1), got {self.attack_fraction}")
        if self.attack_min_len < 1:
            raise ValueError(f"attack_min_len must be >= 1, got {self.attack_min_len}")
        effective_max = min(self.attack_max_len, self.window_size)
        if self.attack_min_len > effective_max:
            raise ValueError(
                f"attack_min_len ({self.attack_min_len}) > effective attack_max_len ({effective_max})"
            )


class SyscallAnomalyDataset(Dataset):
    def __init__(
        self,
        hensel_data: HenselDataset,
        anomaly_cfg: AnomalyDatasetConfig,
        n_samples: int,
        device: torch.device | str = "cpu",
    ) -> None:
        anomaly_cfg.validate()
        self.hensel_data = hensel_data
        self.cfg = anomaly_cfg
        self.n_samples = n_samples
        self.device = torch.device(device)

        rng = torch.Generator()
        rng.manual_seed(anomaly_cfg.seed)

        n_tokens = hensel_data.token_digits.shape[0]
        window = anomaly_cfg.window_size
        n_attack = int(n_samples * anomaly_cfg.attack_fraction)
        n_normal = n_samples - n_attack

        if n_normal < 1 or n_attack < 1:
            raise ValueError(
                "n_samples must produce at least one normal and one attack sample; "
                "increase n_samples or adjust attack_fraction"
            )

        normal_starts = torch.randint(0, n_tokens - window + 1, (n_normal,), generator=rng)
        normal_windows = torch.stack([self._get_window(start, window) for start in normal_starts])

        attack_starts = torch.randint(0, n_tokens - window + 1, (n_attack,), generator=rng)
        attack_windows = torch.stack([self._inject_attack(start, window, rng) for start in attack_starts])

        all_windows = torch.cat([normal_windows, attack_windows], dim=0)
        all_labels = torch.cat(
            [
                torch.zeros(n_normal, dtype=torch.float32),
                torch.ones(n_attack, dtype=torch.float32),
            ]
        )
        perm = torch.randperm(n_samples, generator=rng)
        self.windows = all_windows[perm]
        self.labels = all_labels[perm]

    def _get_window(self, start: int, window: int) -> torch.Tensor:
        n = self.hensel_data.token_digits.shape[0]
        idx = torch.arange(start, start + window) % n
        return self.hensel_data.token_digits[idx].clone()

    def _inject_attack(self, start: int, window: int, rng: torch.Generator) -> torch.Tensor:
        base = self._get_window(start, window).clone()
        n_tokens = self.hensel_data.token_digits.shape[0]

        eff_max = min(self.cfg.attack_max_len, window)
        attack_len = int(torch.randint(self.cfg.attack_min_len, eff_max + 1, (1,), generator=rng).item())
        inject_pos = int(torch.randint(0, window - attack_len + 1, (1,), generator=rng).item())

        region = torch.arange(start + inject_pos, start + inject_pos + attack_len) % n_tokens
        window_labels = self.hensel_data.token_labels[region]
        majority_label = int(window_labels.mode().values.item())
        n_classes = int(self.hensel_data.token_labels.max().item()) + 1
        other_class = (majority_label + 1) % n_classes
        candidate_mask = self.hensel_data.token_labels == other_class
        candidate_idx = candidate_mask.nonzero(as_tuple=True)[0]
        if candidate_idx.numel() == 0:
            candidate_idx = (self.hensel_data.token_labels != majority_label).nonzero(as_tuple=True)[0]
        if candidate_idx.numel() == 0:
            return base

        chosen = candidate_idx[torch.randint(0, candidate_idx.numel(), (attack_len,), generator=rng)]
        base[inject_pos : inject_pos + attack_len] = self.hensel_data.token_digits[chosen].clone()
        return base

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


def build_dataloaders(
    benchmark_cfg: BenchmarkConfig,
    anomaly_cfg: AnomalyDatasetConfig,
    n_train: int,
    n_val: int,
    batch_size: int,
    device: torch.device,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_hensel = generate_clustered_hensel_dataset(benchmark_cfg, device="cpu")
    val_cfg = BenchmarkConfig(
        p=benchmark_cfg.p,
        r=benchmark_cfg.r,
        samples=benchmark_cfg.samples,
        classes=benchmark_cfg.classes,
        tokens_per_class=benchmark_cfg.tokens_per_class,
        seed=benchmark_cfg.seed + 999_999,
        triplets=benchmark_cfg.triplets,
        distance_pairs=benchmark_cfg.distance_pairs,
    )
    val_hensel = generate_clustered_hensel_dataset(val_cfg, device="cpu")

    val_anomaly_cfg = AnomalyDatasetConfig(
        window_size=anomaly_cfg.window_size,
        attack_fraction=anomaly_cfg.attack_fraction,
        attack_min_len=anomaly_cfg.attack_min_len,
        attack_max_len=anomaly_cfg.attack_max_len,
        seed=anomaly_cfg.seed + 1,
    )

    train_ds = SyscallAnomalyDataset(train_hensel, anomaly_cfg, n_train)
    val_ds = SyscallAnomalyDataset(val_hensel, val_anomaly_cfg, n_val)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    return train_loader, val_loader

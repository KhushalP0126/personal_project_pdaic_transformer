"""Realistic anomaly dataset utilities for hardware-like traces."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset


AttackKind = Literal["cross_class", "stuck_at", "burst", "ordering"]


@dataclass(frozen=True)
class RealisticDatasetConfig:
    window_size: int = 32
    attack_fraction: float = 0.005
    idle_fraction: float = 0.70
    attack_min_len: int = 2
    attack_max_len: int = 8
    attack_kinds: tuple[AttackKind, ...] = ("cross_class", "stuck_at", "burst", "ordering")
    seed: int = 20260504

    def validate(self) -> None:
        if not 2 <= self.window_size <= 4096:
            raise ValueError(f"window_size must be in [2, 4096], got {self.window_size}")
        if not 0.0 < self.attack_fraction < 0.5:
            raise ValueError(
                f"attack_fraction={self.attack_fraction} looks wrong. "
                "For hardware realism keep below 0.05."
            )
        if not 0.0 <= self.idle_fraction < 1.0:
            raise ValueError(f"idle_fraction must be in [0, 1), got {self.idle_fraction}")
        if self.attack_min_len < 1:
            raise ValueError(f"attack_min_len must be >= 1, got {self.attack_min_len}")
        eff_max = min(self.attack_max_len, self.window_size)
        if self.attack_min_len > eff_max:
            raise ValueError(
                f"attack_min_len ({self.attack_min_len}) > effective attack_max_len ({eff_max})"
            )
        if not self.attack_kinds:
            raise ValueError("attack_kinds must be non-empty")


def inject_idle_cycles(
    token_digits: torch.Tensor,
    token_labels: torch.Tensor,
    idle_fraction: float,
    p: int,
    r: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a fraction of the stream with an all-zero idle token."""
    del p
    n = token_digits.shape[0]
    n_idle = int(n * idle_fraction)
    perm = torch.randperm(n, generator=rng)
    idle_positions = perm[:n_idle]

    new_digits = token_digits.clone()
    new_labels = token_labels.clone()
    idle_token = torch.zeros(r, dtype=torch.int64, device=token_digits.device)
    new_digits[idle_positions] = idle_token
    new_labels[idle_positions] = -1
    return new_digits, new_labels


def _attack_cross_class(
    base: torch.Tensor,
    inject_pos: int,
    attack_len: int,
    token_digits: torch.Tensor,
    token_labels: torch.Tensor,
    majority_label: int,
    rng: torch.Generator,
) -> torch.Tensor:
    candidate_idx = (token_labels != majority_label).nonzero(as_tuple=True)[0]
    if candidate_idx.numel() == 0:
        warnings.warn(
            "cross_class attack: no cross-class tokens found. Returning unmodified window.",
            RuntimeWarning,
            stacklevel=3,
        )
        return base
    sample_ids = torch.randint(0, candidate_idx.numel(), (attack_len,), generator=rng)
    chosen = candidate_idx[sample_ids]
    base[inject_pos : inject_pos + attack_len] = token_digits[chosen].clone()
    return base


def _attack_stuck_at(
    base: torch.Tensor,
    inject_pos: int,
    attack_len: int,
    token_digits: torch.Tensor,
    rng: torch.Generator,
) -> torch.Tensor:
    pick = int(torch.randint(0, token_digits.shape[0], (1,), generator=rng).item())
    stuck_token = token_digits[pick].clone()
    base[inject_pos : inject_pos + attack_len] = stuck_token.unsqueeze(0).expand(attack_len, -1)
    return base


def _attack_burst(
    base: torch.Tensor,
    inject_pos: int,
    attack_len: int,
    token_digits: torch.Tensor,
    token_labels: torch.Tensor,
    rng: torch.Generator,
) -> torch.Tensor:
    n_classes = int(token_labels[token_labels >= 0].max().item()) + 1
    target_class = int(torch.randint(0, n_classes, (1,), generator=rng).item())
    candidate_idx = (token_labels == target_class).nonzero(as_tuple=True)[0]
    if candidate_idx.numel() == 0:
        return base
    sample_ids = torch.randint(0, candidate_idx.numel(), (attack_len,), generator=rng)
    chosen = candidate_idx[sample_ids]
    base[inject_pos : inject_pos + attack_len] = token_digits[chosen].clone()
    return base


def _attack_ordering(
    base: torch.Tensor,
    inject_pos: int,
    attack_len: int,
    rng: torch.Generator,
) -> torch.Tensor:
    segment = base[inject_pos : inject_pos + attack_len].clone()
    perm = torch.randperm(attack_len, generator=rng)
    base[inject_pos : inject_pos + attack_len] = segment[perm]
    return base


class RealisticBusDataset(Dataset):
    """Idle-heavy anomaly dataset with multiple attack archetypes."""

    def __init__(
        self,
        hensel_data: object,
        cfg: RealisticDatasetConfig,
        n_samples: int,
    ) -> None:
        cfg.validate()
        if cfg.attack_fraction > 0.05:
            warnings.warn(
                f"attack_fraction={cfg.attack_fraction:.3f} is above 5%. "
                "This is unrealistically high for hardware bus anomaly detection.",
                UserWarning,
                stacklevel=2,
            )

        self.hensel_data = hensel_data
        self.cfg = cfg
        self.n_samples = n_samples

        rng = torch.Generator()
        rng.manual_seed(cfg.seed)

        token_digits, token_labels = inject_idle_cycles(
            hensel_data.token_digits,
            hensel_data.token_labels,
            cfg.idle_fraction,
            p=int(hensel_data.token_digits.max().item()) + 1,
            r=hensel_data.token_digits.shape[1],
            rng=rng,
        )

        n_tokens = token_digits.shape[0]
        window = cfg.window_size
        if n_tokens < window:
            raise ValueError(
                f"window_size ({window}) > token stream length ({n_tokens}) after idle injection."
            )

        n_attack = max(1, int(n_samples * cfg.attack_fraction))
        n_normal = n_samples - n_attack
        if n_normal < 1:
            raise ValueError("n_samples too small for the given attack_fraction")

        normal_starts = torch.randint(0, n_tokens - window + 1, (n_normal,), generator=rng)
        normal_windows = torch.stack([token_digits[s : s + window].clone() for s in normal_starts])

        attack_starts = torch.randint(0, n_tokens - window + 1, (n_attack,), generator=rng)
        attack_windows = []
        self.attack_kind_counts: dict[str, int] = {k: 0 for k in cfg.attack_kinds}

        for start in attack_starts:
            win, kind = self._inject_attack(
                int(start.item()), window, token_digits, token_labels, rng
            )
            attack_windows.append(win)
            self.attack_kind_counts[kind] += 1

        all_windows = torch.cat([normal_windows, torch.stack(attack_windows)], dim=0)
        all_labels = torch.cat(
            [
                torch.zeros(n_normal, dtype=torch.float32),
                torch.ones(n_attack, dtype=torch.float32),
            ]
        )
        perm = torch.randperm(n_samples, generator=rng)
        self.windows = all_windows[perm]
        self.labels = all_labels[perm]

        n_pos = int(self.labels.sum().item())
        n_neg = n_samples - n_pos
        self.pos_weight: float = n_neg / max(1, n_pos)
        print(
            f"RealisticBusDataset: {n_samples} samples | normal={n_normal} attack={n_attack} "
            f"(pos_weight={self.pos_weight:.1f}) | idle_fraction={cfg.idle_fraction:.2f} "
            f"| attack_kinds={self.attack_kind_counts}"
        )

    def _inject_attack(
        self,
        start: int,
        window: int,
        token_digits: torch.Tensor,
        token_labels: torch.Tensor,
        rng: torch.Generator,
    ) -> tuple[torch.Tensor, AttackKind]:
        base = token_digits[start : start + window].clone()

        eff_max = min(self.cfg.attack_max_len, window)
        attack_len = int(torch.randint(self.cfg.attack_min_len, eff_max + 1, (1,), generator=rng).item())
        inject_pos = int(torch.randint(0, window - attack_len + 1, (1,), generator=rng).item())

        kind_idx = int(torch.randint(0, len(self.cfg.attack_kinds), (1,), generator=rng).item())
        kind: AttackKind = self.cfg.attack_kinds[kind_idx]

        region = torch.arange(start + inject_pos, start + inject_pos + attack_len)
        window_labels = token_labels[region]
        valid_mask = window_labels >= 0
        if valid_mask.sum() == 0:
            majority_label = 0
        else:
            majority_label = int(window_labels[valid_mask].mode().values.item())

        if kind == "cross_class":
            base = _attack_cross_class(
                base, inject_pos, attack_len, token_digits, token_labels, majority_label, rng
            )
        elif kind == "stuck_at":
            base = _attack_stuck_at(base, inject_pos, attack_len, token_digits, rng)
        elif kind == "burst":
            base = _attack_burst(base, inject_pos, attack_len, token_digits, token_labels, rng)
        elif kind == "ordering":
            base = _attack_ordering(base, inject_pos, attack_len, rng)

        return base, kind

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


def make_weighted_loss(
    dataset: RealisticBusDataset,
    p: int,
    alpha: float = 0.5,
    margin_pos: float = 0.1,
    margin_neg: float = 0.5,
    max_pairs: int = 4096,
):
    from .losses import AnomalyLoss

    print(f"make_weighted_loss: pos_weight={dataset.pos_weight:.2f} (n_neg/n_pos)")
    return AnomalyLoss(
        p=p,
        alpha=alpha,
        pos_weight=dataset.pos_weight,
        margin_pos=margin_pos,
        margin_neg=margin_neg,
        max_pairs=max_pairs,
    )

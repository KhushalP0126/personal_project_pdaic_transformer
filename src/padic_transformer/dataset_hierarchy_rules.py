"""Synthetic anomaly dataset that tests subtree-stay rule learning."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
from torch.utils.data import Dataset

ATTACK_RETRY_LIMIT = 12


@dataclass(frozen=True)
class HierarchyRuleDatasetConfig:
    window_size: int = 32
    attack_fraction: float = 0.3
    subtree_depth: int = 2
    stay_steps: int = 4
    attack_tokens: int = 1
    seed: int = 20260504

    def validate(self, r: int) -> None:
        if not 2 <= self.window_size <= 4096:
            raise ValueError(f"window_size must be in [2, 4096], got {self.window_size}")
        if not 0.0 < self.attack_fraction < 1.0:
            raise ValueError(f"attack_fraction must be in (0, 1), got {self.attack_fraction}")
        if not 1 <= self.subtree_depth <= r:
            raise ValueError(f"subtree_depth must be in [1, {r}], got {self.subtree_depth}")
        if not 1 <= self.stay_steps <= self.window_size:
            raise ValueError(f"stay_steps must be in [1, {self.window_size}], got {self.stay_steps}")
        if not 1 <= self.attack_tokens <= self.window_size:
            raise ValueError(f"attack_tokens must be in [1, {self.window_size}], got {self.attack_tokens}")


class HierarchyRuleDataset(Dataset):
    """Windows that obey or violate a p-adic subtree persistence rule."""

    def __init__(
        self,
        hensel_data: object,
        cfg: HierarchyRuleDatasetConfig,
        n_samples: int,
    ) -> None:
        cfg.validate(hensel_data.token_digits.shape[1])
        self.hensel_data = hensel_data
        self.cfg = cfg
        self.n_samples = n_samples

        rng = torch.Generator()
        rng.manual_seed(cfg.seed)

        self._prefix_groups = self._build_prefix_groups()
        if len(self._prefix_groups) < 2:
            raise ValueError("HierarchyRuleDataset requires at least two subtree groups")

        n_attack = int(n_samples * cfg.attack_fraction)
        n_normal = n_samples - n_attack
        if n_normal < 1 or n_attack < 1:
            raise ValueError("n_samples must produce at least one normal and one attack sample")

        normal_windows = torch.stack([self._sample_normal_window(rng) for _ in range(n_normal)])
        attack_results = [self._sample_attack_window(rng) for _ in range(n_attack)]
        attack_windows = torch.stack([window for window, _ in attack_results])
        attack_labels = torch.tensor(
            [float(success) for _, success in attack_results],
            dtype=torch.float32,
        )
        successful_attacks = int(attack_labels.sum().item())
        if successful_attacks == 0:
            raise ValueError(
                "failed to generate any non-trivial hierarchy-rule attacks; "
                "reduce subtree_depth or increase class diversity"
            )
        if successful_attacks < n_attack:
            warnings.warn(
                f"HierarchyRuleDataset: generated {successful_attacks}/{n_attack} requested attacks; "
                "failed attack attempts were kept as normal samples.",
                RuntimeWarning,
                stacklevel=2,
            )
        all_windows = torch.cat([normal_windows, attack_windows], dim=0)
        all_labels = torch.cat(
            [
                torch.zeros(n_normal, dtype=torch.float32),
                attack_labels,
            ]
        )
        perm = torch.randperm(n_samples, generator=rng)
        self.windows = all_windows[perm]
        self.labels = all_labels[perm]

    def _build_prefix_groups(self) -> list[torch.Tensor]:
        digits = self.hensel_data.token_digits
        depth = self.cfg.subtree_depth
        groups: dict[tuple[int, ...], list[int]] = {}
        for idx, row in enumerate(digits):
            prefix = tuple(int(v) for v in row[:depth].tolist())
            groups.setdefault(prefix, []).append(idx)
        valid = [torch.tensor(indices, dtype=torch.int64) for indices in groups.values() if indices]
        if len(valid) < 2:
            warnings.warn(
                "HierarchyRuleDataset: too few subtree groups at the requested depth; "
                "consider reducing subtree_depth or increasing class diversity.",
                RuntimeWarning,
                stacklevel=2,
            )
        return valid

    def _sample_group_tokens(self, group_idx: int, count: int, rng: torch.Generator) -> torch.Tensor:
        group = self._prefix_groups[group_idx]
        sample_ids = torch.randint(0, group.numel(), (count,), generator=rng)
        token_ids = group[sample_ids]
        return self.hensel_data.token_digits[token_ids].clone()

    def _sample_normal_window(self, rng: torch.Generator) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        remaining = self.cfg.window_size
        while remaining > 0:
            group_idx = int(torch.randint(0, len(self._prefix_groups), (1,), generator=rng).item())
            segment_len = min(self.cfg.stay_steps, remaining)
            pieces.append(self._sample_group_tokens(group_idx, segment_len, rng))
            remaining -= segment_len
        return torch.cat(pieces, dim=0)

    def _sample_attack_window(self, rng: torch.Generator) -> tuple[torch.Tensor, bool]:
        base = self._sample_normal_window(rng)
        max_start = self.cfg.window_size - self.cfg.attack_tokens
        for _ in range(ATTACK_RETRY_LIMIT):
            start = int(torch.randint(0, max_start + 1, (1,), generator=rng).item())
            before = base[start : start + self.cfg.attack_tokens].clone()
            prefix = tuple(int(v) for v in before[0, : self.cfg.subtree_depth].tolist())
            candidate_groups = [
                idx
                for idx, group in enumerate(self._prefix_groups)
                if tuple(
                    int(v)
                    for v in self.hensel_data.token_digits[
                        group[0], : self.cfg.subtree_depth
                    ].tolist()
                )
                != prefix
            ]
            if not candidate_groups:
                break
            group_pick = int(torch.randint(0, len(candidate_groups), (1,), generator=rng).item())
            replacement = self._sample_group_tokens(
                candidate_groups[group_pick],
                self.cfg.attack_tokens,
                rng,
            )
            if torch.equal(replacement, before):
                continue
            base[start : start + self.cfg.attack_tokens] = replacement
            return base, True

        warnings.warn(
            "HierarchyRuleDataset: failed to create a non-trivial subtree-jump anomaly after several attempts; "
            "returning the original window with a normal label.",
            RuntimeWarning,
            stacklevel=2,
        )
        return base, False

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]

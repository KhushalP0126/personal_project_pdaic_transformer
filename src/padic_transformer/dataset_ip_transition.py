"""Transition-based synthetic IPv4 prefix anomaly datasets.

This module adds a harder controlled task than the simple prefix-jump dataset.
Normal and anomalous windows use the same prefix vocabulary, so a model cannot
solve the task by only noticing that an unfamiliar prefix appeared. The label is
instead determined by whether the sequence follows a legal prefix-transition
rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset

from .dataset_ip import P, R, int_to_binary_digits


IPTransitionAttackKind = Literal["illegal_transition", "cycle_reversal", "skip_transition"]


@dataclass(frozen=True)
class IPPrefixTransitionDatasetConfig:
    """Configuration for transition-based IP-prefix anomaly windows.

    Prefixes are split into groups. A normal window stays inside one group and
    advances through that group's prefix cycle with step +1. Attack windows use
    the same prefixes but violate the transition rule by reversing the cycle,
    skipping ahead, or inserting a transition into a different group.
    """

    window_size: int = 16
    attack_fraction: float = 0.3
    prefix_len: int = 24
    num_prefixes: int = 32
    num_groups: int = 4
    attack_kinds: tuple[IPTransitionAttackKind, ...] = (
        "illegal_transition",
        "cycle_reversal",
        "skip_transition",
    )
    seed: int = 20260504

    def validate(self) -> None:
        if not 4 <= self.window_size <= 4096:
            raise ValueError(f"window_size must be in [4, 4096], got {self.window_size}")
        if not 0.0 < self.attack_fraction < 1.0:
            raise ValueError(f"attack_fraction must be in (0, 1), got {self.attack_fraction}")
        if not 1 <= self.prefix_len < R:
            raise ValueError(f"prefix_len must be in [1, {R - 1}], got {self.prefix_len}")
        if self.num_prefixes < 4:
            raise ValueError("num_prefixes must be at least 4")
        if self.num_prefixes > (1 << self.prefix_len):
            raise ValueError("num_prefixes cannot exceed the prefix address space")
        if not 1 <= self.num_groups <= self.num_prefixes:
            raise ValueError("num_groups must be in [1, num_prefixes]")
        if self.num_prefixes % self.num_groups != 0:
            raise ValueError("num_prefixes must be divisible by num_groups")
        if self.num_prefixes // self.num_groups < 3:
            raise ValueError("each group must contain at least 3 prefixes")
        valid_kinds = {"illegal_transition", "cycle_reversal", "skip_transition"}
        unknown = set(self.attack_kinds) - valid_kinds
        if unknown:
            raise ValueError(f"unknown attack kinds: {sorted(unknown)}")
        if not self.attack_kinds:
            raise ValueError("attack_kinds must be non-empty")


class IPPrefixTransitionAnomalyDataset(Dataset):
    """Windows whose anomalies are violations of prefix-transition rules.

    Compared with IPPrefixAnomalyDataset, this task is harder because all labels
    draw addresses from the same prefix vocabulary. The anomaly signal is mostly
    sequential: a local transition breaks the legal group cycle.
    """

    p: int = P
    r: int = R

    def __init__(self, cfg: IPPrefixTransitionDatasetConfig, n_samples: int) -> None:
        cfg.validate()
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")

        self.cfg = cfg
        self.n_samples = n_samples
        self.group_size = cfg.num_prefixes // cfg.num_groups

        rng = torch.Generator()
        rng.manual_seed(cfg.seed)

        self.prefix_values = self._draw_prefix_values(rng)
        self.groups = [
            list(range(group_id * self.group_size, (group_id + 1) * self.group_size))
            for group_id in range(cfg.num_groups)
        ]

        n_attack = int(n_samples * cfg.attack_fraction)
        n_normal = n_samples - n_attack
        if n_attack < 1 or n_normal < 1:
            raise ValueError("n_samples must produce at least one normal and one attack sample")

        normal = [self._sample_normal_window(rng) for _ in range(n_normal)]
        attacks = [self._sample_attack_window(rng) for _ in range(n_attack)]

        all_windows = torch.stack([w for w, _, _ in normal + attacks])
        all_prefix_ids = torch.stack([pids for _, pids, _ in normal + attacks])
        all_labels = torch.cat(
            [
                torch.zeros(n_normal, dtype=torch.float32),
                torch.ones(n_attack, dtype=torch.float32),
            ]
        )
        all_kinds = [kind for _, _, kind in normal + attacks]

        perm = torch.randperm(n_samples, generator=rng)
        self.windows = all_windows[perm]
        self.window_prefix_ids = all_prefix_ids[perm]
        self.labels = all_labels[perm]
        self.attack_kinds = [all_kinds[int(idx.item())] for idx in perm]

        self.attack_kind_counts = {kind: 0 for kind in cfg.attack_kinds}
        for kind, label in zip(self.attack_kinds, self.labels):
            if int(label.item()) == 1:
                self.attack_kind_counts[kind] = self.attack_kind_counts.get(kind, 0) + 1

    def _draw_prefix_values(self, rng: torch.Generator) -> list[int]:
        values: set[int] = set()
        high = 1 << self.cfg.prefix_len
        while len(values) < self.cfg.num_prefixes:
            values.add(int(torch.randint(0, high, (1,), generator=rng).item()))
        return list(values)

    def _sample_ip_from_prefix(self, prefix_id: int, rng: torch.Generator) -> torch.Tensor:
        prefix_value = self.prefix_values[prefix_id]
        host_bits = R - self.cfg.prefix_len
        host = int(torch.randint(0, 1 << host_bits, (1,), generator=rng).item())
        value = (prefix_value << host_bits) | host
        return int_to_binary_digits(value, width=R)

    def _sample_group_id(self, rng: torch.Generator, exclude: int | None = None) -> int:
        if exclude is None:
            return int(torch.randint(0, self.cfg.num_groups, (1,), generator=rng).item())
        raw = int(torch.randint(0, self.cfg.num_groups - 1, (1,), generator=rng).item())
        return raw + (1 if raw >= exclude else 0)

    def _normal_prefix_sequence(self, rng: torch.Generator) -> tuple[torch.Tensor, int]:
        group_id = self._sample_group_id(rng)
        group = self.groups[group_id]
        start = int(torch.randint(0, self.group_size, (1,), generator=rng).item())
        ids = [group[(start + pos) % self.group_size] for pos in range(self.cfg.window_size)]
        return torch.tensor(ids, dtype=torch.int64), group_id

    def _materialize_window(self, prefix_ids: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        return torch.stack([self._sample_ip_from_prefix(int(prefix_id.item()), rng) for prefix_id in prefix_ids])

    def _sample_normal_window(self, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, str]:
        prefix_ids, _ = self._normal_prefix_sequence(rng)
        return self._materialize_window(prefix_ids, rng), prefix_ids, "normal"

    def _sample_attack_window(self, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, str]:
        prefix_ids, group_id = self._normal_prefix_sequence(rng)
        kind_id = int(torch.randint(0, len(self.cfg.attack_kinds), (1,), generator=rng).item())
        kind = self.cfg.attack_kinds[kind_id]
        pos = int(torch.randint(1, self.cfg.window_size, (1,), generator=rng).item())

        if kind == "illegal_transition":
            other_group_id = self._sample_group_id(rng, exclude=group_id)
            other_group = self.groups[other_group_id]
            replacement = other_group[int(torch.randint(0, self.group_size, (1,), generator=rng).item())]
            prefix_ids[pos] = replacement
        elif kind == "cycle_reversal":
            group = self.groups[group_id]
            previous_idx = group.index(int(prefix_ids[pos - 1].item()))
            prefix_ids[pos] = group[(previous_idx - 1) % self.group_size]
        elif kind == "skip_transition":
            group = self.groups[group_id]
            previous_idx = group.index(int(prefix_ids[pos - 1].item()))
            prefix_ids[pos] = group[(previous_idx + 2) % self.group_size]
        else:
            raise RuntimeError(f"unhandled attack kind: {kind}")

        return self._materialize_window(prefix_ids, rng), prefix_ids, kind

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]

"""Synthetic IPv4 prefix anomaly datasets for 2-adic attention experiments."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from typing import Literal

import torch
from torch.utils.data import Dataset


IPAttackKind = Literal["prefix_jump", "spoofed_prefix", "route_leak"]
P = 2
R = 32


def ipv4_to_int(addr: str) -> int:
    """Convert an IPv4 string to its unsigned 32-bit integer value."""
    return int(ipaddress.IPv4Address(addr))


def int_to_ipv4(value: int) -> str:
    """Convert an unsigned 32-bit integer value to dotted IPv4 notation."""
    if not 0 <= value < (1 << R):
        raise ValueError(f"IPv4 integer must be in [0, 2**32), got {value}")
    return str(ipaddress.IPv4Address(value))


def int_to_binary_digits(value: int, width: int = R) -> torch.Tensor:
    """Return MSB-first binary digits.

    Existing p-adic attention metrics compare shared prefixes starting at digit
    index 0. IPv4 network prefixes are leftmost bits, so this dataset stores
    addresses MSB-first instead of Hensel's usual least-significant-first order.
    """
    if not 1 <= width <= 63:
        raise ValueError(f"width must be in [1, 63], got {width}")
    if not 0 <= value < (1 << width):
        raise ValueError(f"value must be in [0, 2**width), got {value}")
    shifts = torch.arange(width - 1, -1, -1, dtype=torch.int64)
    return ((int(value) >> shifts) & 1).to(torch.int64)


def ipv4_to_binary_digits(addr: str) -> torch.Tensor:
    """Convert an IPv4 string to 32 MSB-first binary digits."""
    return int_to_binary_digits(ipv4_to_int(addr), width=R)


def binary_digits_to_int(digits: torch.Tensor) -> int:
    """Pack MSB-first binary digits into an integer."""
    arr = torch.as_tensor(digits, dtype=torch.int64)
    if arr.ndim != 1:
        raise ValueError("digits must be a 1D tensor")
    if bool(torch.any((arr < 0) | (arr > 1)).item()):
        raise ValueError("binary digits must be 0 or 1")
    width = arr.numel()
    shifts = torch.arange(width - 1, -1, -1, dtype=torch.int64, device=arr.device)
    return int((arr * (1 << shifts)).sum().item())


@dataclass(frozen=True)
class IPPrefixDatasetConfig:
    window_size: int = 32
    attack_fraction: float = 0.3
    prefix_len: int = 24
    num_prefixes: int = 16
    attack_min_len: int = 1
    attack_max_len: int = 4
    attack_kinds: tuple[IPAttackKind, ...] = (
        "prefix_jump",
        "spoofed_prefix",
        "route_leak",
    )
    seed: int = 20260504

    def validate(self) -> None:
        if not 2 <= self.window_size <= 4096:
            raise ValueError(f"window_size must be in [2, 4096], got {self.window_size}")
        if not 0.0 < self.attack_fraction < 1.0:
            raise ValueError(f"attack_fraction must be in (0, 1), got {self.attack_fraction}")
        if not 1 <= self.prefix_len < R:
            raise ValueError(f"prefix_len must be in [1, {R - 1}], got {self.prefix_len}")
        if self.num_prefixes < 2:
            raise ValueError("num_prefixes must be at least 2")
        if self.num_prefixes > (1 << self.prefix_len):
            raise ValueError("num_prefixes cannot exceed the prefix address space")
        if self.attack_min_len < 1:
            raise ValueError(f"attack_min_len must be >= 1, got {self.attack_min_len}")
        if self.attack_max_len > self.window_size:
            raise ValueError(
                f"attack_max_len ({self.attack_max_len}) cannot exceed "
                f"window_size ({self.window_size})"
            )
        if self.attack_min_len > self.attack_max_len:
            raise ValueError(
                f"attack_min_len ({self.attack_min_len}) > attack_max_len ({self.attack_max_len})"
            )
        valid_kinds = {"prefix_jump", "spoofed_prefix", "route_leak"}
        unknown = set(self.attack_kinds) - valid_kinds
        if unknown:
            raise ValueError(f"unknown attack kinds: {sorted(unknown)}")
        if not self.attack_kinds:
            raise ValueError("attack_kinds must be non-empty")
        if "spoofed_prefix" in self.attack_kinds and self.num_prefixes >= (1 << self.prefix_len):
            raise ValueError("spoofed_prefix attacks require at least one unused prefix")


class IPPrefixAnomalyDataset(Dataset):
    """Windows of IPv4 addresses with prefix-stable normal traffic and prefix-jump anomalies."""

    p: int = P
    r: int = R

    def __init__(self, cfg: IPPrefixDatasetConfig, n_samples: int) -> None:
        cfg.validate()
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")

        self.cfg = cfg
        self.n_samples = n_samples

        rng = torch.Generator()
        rng.manual_seed(cfg.seed)

        self.prefix_values = self._draw_prefix_values(rng)
        self.spoof_prefix_values = (
            self._draw_spoof_prefix_values(rng)
            if "spoofed_prefix" in cfg.attack_kinds
            else []
        )

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
            candidate = int(torch.randint(0, high, (1,), generator=rng).item())
            values.add(candidate)
        return list(values)

    def _draw_spoof_prefix_values(self, rng: torch.Generator) -> list[int]:
        values: set[int] = set()
        used = set(self.prefix_values)
        high = 1 << self.cfg.prefix_len
        target_count = max(1, min(self.cfg.num_prefixes, high - len(used)))
        while len(values) < target_count:
            candidate = int(torch.randint(0, high, (1,), generator=rng).item())
            if candidate not in used:
                values.add(candidate)
        return list(values)

    def _sample_ip_from_prefix(self, prefix_value: int, rng: torch.Generator) -> torch.Tensor:
        host_bits = R - self.cfg.prefix_len
        host = int(torch.randint(0, 1 << host_bits, (1,), generator=rng).item())
        value = (prefix_value << host_bits) | host
        return int_to_binary_digits(value, width=R)

    def _sample_prefix_id(self, rng: torch.Generator, exclude: int | None = None) -> int:
        if exclude is None:
            return int(torch.randint(0, self.cfg.num_prefixes, (1,), generator=rng).item())
        raw = int(torch.randint(0, self.cfg.num_prefixes - 1, (1,), generator=rng).item())
        return raw + (1 if raw >= exclude else 0)

    def _sample_normal_window(
        self,
        rng: torch.Generator,
        prefix_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        if prefix_id is None:
            prefix_id = self._sample_prefix_id(rng)
        prefix_value = self.prefix_values[prefix_id]
        rows = [
            self._sample_ip_from_prefix(prefix_value, rng)
            for _ in range(self.cfg.window_size)
        ]
        prefix_ids = torch.full((self.cfg.window_size,), prefix_id, dtype=torch.int64)
        return torch.stack(rows), prefix_ids, "normal"

    def _sample_attack_window(self, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, str]:
        base_prefix = self._sample_prefix_id(rng)
        window, prefix_ids, _ = self._sample_normal_window(rng, prefix_id=base_prefix)
        attack_len = int(
            torch.randint(
                self.cfg.attack_min_len,
                self.cfg.attack_max_len + 1,
                (1,),
                generator=rng,
            ).item()
        )
        start = int(
            torch.randint(0, self.cfg.window_size - attack_len + 1, (1,), generator=rng).item()
        )
        kind_id = int(torch.randint(0, len(self.cfg.attack_kinds), (1,), generator=rng).item())
        kind = self.cfg.attack_kinds[kind_id]

        if kind == "prefix_jump":
            other_prefix = self._sample_prefix_id(rng, exclude=base_prefix)
            for pos in range(start, start + attack_len):
                window[pos] = self._sample_ip_from_prefix(self.prefix_values[other_prefix], rng)
                prefix_ids[pos] = other_prefix
        elif kind == "spoofed_prefix":
            spoof_id = int(torch.randint(0, len(self.spoof_prefix_values), (1,), generator=rng).item())
            spoof_prefix = self.spoof_prefix_values[spoof_id]
            for pos in range(start, start + attack_len):
                window[pos] = self._sample_ip_from_prefix(spoof_prefix, rng)
                prefix_ids[pos] = -1
        elif kind == "route_leak":
            for pos in range(start, start + attack_len):
                other_prefix = self._sample_prefix_id(rng, exclude=base_prefix)
                window[pos] = self._sample_ip_from_prefix(self.prefix_values[other_prefix], rng)
                prefix_ids[pos] = other_prefix
        else:
            raise RuntimeError(f"unhandled attack kind: {kind}")

        return window, prefix_ids, kind

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]

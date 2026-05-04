"""Validated configuration objects for p-adic benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


def is_prime(value: int) -> bool:
    """Return True when value is a small positive prime."""
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    factor = 3
    while factor * factor <= value:
        if value % factor == 0:
            return False
        factor += 2
    return True


@dataclass(frozen=True)
class BenchmarkConfig:
    """Bounded benchmark configuration.

    The bounds keep accidental cloud runs from allocating unreasonably large tensors.
    They can be raised intentionally after the first GPU profile.
    """

    p: int
    r: int
    samples: int = 4096
    classes: int = 16
    tokens_per_class: int = 64
    seed: int = 20260504
    triplets: int = 20000
    distance_pairs: int = 200000

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not is_prime(self.p):
            raise ValueError(f"p must be prime, got {self.p}")
        if not 1 <= self.r <= 64:
            raise ValueError(f"r must be in [1, 64], got {self.r}")
        if not 16 <= self.samples <= 1_000_000:
            raise ValueError(f"samples must be in [16, 1000000], got {self.samples}")
        if not 2 <= self.classes <= 4096:
            raise ValueError(f"classes must be in [2, 4096], got {self.classes}")
        if not 1 <= self.tokens_per_class <= 100_000:
            raise ValueError(
                f"tokens_per_class must be in [1, 100000], got {self.tokens_per_class}"
            )
        if not 1 <= self.triplets <= 5_000_000:
            raise ValueError(f"triplets must be in [1, 5000000], got {self.triplets}")
        if not 1 <= self.distance_pairs <= 10_000_000:
            raise ValueError(
                f"distance_pairs must be in [1, 10000000], got {self.distance_pairs}"
            )

"""p-adic transformer phase-1 benchmark primitives."""

from .config import BenchmarkConfig
from .hensel import carry_left_add, digits_to_int64, int64_to_digits, shared_prefix_valuation
from .ultrametric import generate_clustered_hensel_dataset, ultrametric_violation_rate

__all__ = [
    "BenchmarkConfig",
    "carry_left_add",
    "digits_to_int64",
    "generate_clustered_hensel_dataset",
    "int64_to_digits",
    "shared_prefix_valuation",
    "ultrametric_violation_rate",
]

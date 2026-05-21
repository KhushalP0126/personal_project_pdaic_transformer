"""p-adic transformer phase-1 benchmark and training primitives."""

from .config import BenchmarkConfig
from .dataset import AnomalyDatasetConfig, SyscallAnomalyDataset, build_dataloaders
from .hensel import (
    carry_left_add,
    digits_to_int64,
    formal_power_series_coefficients,
    int64_to_digits,
    shared_prefix_valuation,
)
from .losses import AnomalyLoss, PadicContrastiveLoss
from .model import AnomalyHead, HenselEmbedding, PadicAnomalyDetector, PadicTransformerEncoder
from .padic_attention import (
    AnomalyHead as PadicAttentionAnomalyHead,
    HenselEmbedding as PadicAttentionHenselEmbedding,
    PadicAttentionAnomalyDetector,
    PadicAttentionEncoder,
    PadicAttentionHead,
    PadicMultiHeadAttention,
    PadicTransformerLayer,
    SoftPadicValuation,
    compare_hard_soft_valuation,
)
from .training import TrainConfig, train
from .ultrametric import generate_clustered_hensel_dataset, ultrametric_violation_rate

__all__ = [
    "BenchmarkConfig",
    "AnomalyDatasetConfig",
    "AnomalyHead",
    "AnomalyLoss",
    "PadicAttentionAnomalyDetector",
    "PadicAttentionEncoder",
    "PadicAttentionHead",
    "PadicAttentionHenselEmbedding",
    "PadicAttentionAnomalyHead",
    "PadicMultiHeadAttention",
    "carry_left_add",
    "digits_to_int64",
    "formal_power_series_coefficients",
    "generate_clustered_hensel_dataset",
    "HenselEmbedding",
    "int64_to_digits",
    "PadicAnomalyDetector",
    "PadicContrastiveLoss",
    "PadicTransformerEncoder",
    "PadicTransformerLayer",
    "SyscallAnomalyDataset",
    "TrainConfig",
    "SoftPadicValuation",
    "build_dataloaders",
    "compare_hard_soft_valuation",
    "shared_prefix_valuation",
    "train",
    "ultrametric_violation_rate",
]

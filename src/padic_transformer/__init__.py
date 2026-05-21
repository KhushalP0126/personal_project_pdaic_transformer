"""p-adic transformer phase-1 benchmark and training primitives."""

from .config import BenchmarkConfig
from .dataset import AnomalyDatasetConfig, SyscallAnomalyDataset, build_dataloaders
from .dataset_hierarchy_rules import HierarchyRuleDataset, HierarchyRuleDatasetConfig
from .dataset_realistic import RealisticBusDataset, RealisticDatasetConfig, make_weighted_loss
from .hensel import (
    carry_left_add,
    digits_to_int64,
    formal_power_series_coefficients,
    int64_to_digits,
    shared_prefix_valuation,
)
from .losses import AnomalyLoss, PadicContrastiveLoss
from .model import AnomalyHead, HenselEmbedding, PadicAnomalyDetector, PadicTransformerEncoder
from .model_fixes import (
    HenselEmbeddingWithPosition,
    StreamingWindowScorer,
    compute_diversity_regularization,
    log_temperature_health,
    quantize_dynamic_model,
)
from .metrics import binary_auroc
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
    "binary_auroc",
    "HierarchyRuleDataset",
    "HierarchyRuleDatasetConfig",
    "RealisticBusDataset",
    "RealisticDatasetConfig",
    "compute_diversity_regularization",
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
    "HenselEmbeddingWithPosition",
    "int64_to_digits",
    "PadicAnomalyDetector",
    "PadicContrastiveLoss",
    "PadicTransformerEncoder",
    "PadicTransformerLayer",
    "quantize_dynamic_model",
    "StreamingWindowScorer",
    "SyscallAnomalyDataset",
    "TrainConfig",
    "log_temperature_health",
    "make_weighted_loss",
    "SoftPadicValuation",
    "build_dataloaders",
    "compare_hard_soft_valuation",
    "shared_prefix_valuation",
    "train",
    "ultrametric_violation_rate",
]

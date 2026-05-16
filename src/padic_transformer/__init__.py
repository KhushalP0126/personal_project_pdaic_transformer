"""p-adic transformer phase-1 benchmark and training primitives."""

from .config import BenchmarkConfig
from .dataset import AnomalyDatasetConfig, SyscallAnomalyDataset, build_dataloaders
from .hensel import carry_left_add, digits_to_int64, int64_to_digits, shared_prefix_valuation
from .losses import AnomalyLoss, PadicContrastiveLoss
from .model import AnomalyHead, HenselEmbedding, PadicAnomalyDetector, PadicTransformerEncoder
from .training import TrainConfig, train
from .ultrametric import generate_clustered_hensel_dataset, ultrametric_violation_rate

__all__ = [
    "BenchmarkConfig",
    "AnomalyDatasetConfig",
    "AnomalyHead",
    "AnomalyLoss",
    "carry_left_add",
    "digits_to_int64",
    "generate_clustered_hensel_dataset",
    "HenselEmbedding",
    "int64_to_digits",
    "PadicAnomalyDetector",
    "PadicContrastiveLoss",
    "PadicTransformerEncoder",
    "SyscallAnomalyDataset",
    "TrainConfig",
    "build_dataloaders",
    "shared_prefix_valuation",
    "train",
    "ultrametric_violation_rate",
]

"""
experiment_controller.py
------------------------
Comparative Experiment Runner for the p-adic Transformer research harness.

Implements:
  1. Chunked nearest_center_accuracy   — OOM-safe at N=16384, C=32
  2. Device-aware distance kernels     — no blocking .item()/.cpu() in hot path
  3. PadicLinear / FP32Linear toggle   — drop-in nn.Module replacement
  4. MetricLogger                      — F1, latency, bit-flip resilience
  5. ExperimentController              — orchestrates a single sweep run
  6. gpu_smoke_test()                  — fast pre-flight before the full sweep

Usage (quick smoke test):
    python experiment_controller.py --smoke-test --device cuda
    python experiment_controller.py --smoke-test --device cpu

Full sweep entry-point:
    from experiment_controller import ExperimentController, SweepConfig
    ctrl = ExperimentController(SweepConfig(...))
    results = ctrl.run()
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Local imports — adjust sys.path when running outside the package
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from padic_transformer.hensel import (  # noqa: E402
    digits_to_int64,
    int64_to_digits,
    shared_prefix_valuation,
)
from padic_transformer.config import BenchmarkConfig  # noqa: E402


# ===========================================================================
# 1. CHUNKED nearest_center_accuracy
#    Replaces the O(N * C * r) eager allocation with an O(chunk * C * r) peak.
# ===========================================================================

def nearest_center_accuracy_chunked(
    digits: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    chunk_size: int = 512,
) -> float:
    """
    Classify tokens by nearest p-adic center, avoiding OOM via row-chunking.

    Peak memory: O(chunk_size * C * r)  instead of O(N * C * r).

    At N=16384, C=32, r=32, fp32: eager = 16384*32*32*4 = 67 MB per tensor.
    With chunk_size=512 that drops to ~2 MB per chunk.

    Args:
        digits:     [N, r]  int64 Hensel digits
        labels:     [N]     int64 ground-truth class indices
        centers:    [C, r]  int64 center digits
        chunk_size: rows of `digits` processed per iteration

    Returns:
        Scalar accuracy in [0, 1].
    """
    if digits.ndim != 2 or centers.ndim != 2:
        raise ValueError("digits and centers must both be 2-D")
    if digits.shape[1] != centers.shape[1]:
        raise ValueError("digits and centers must share precision axis r")
    if labels.shape[0] != digits.shape[0]:
        raise ValueError("labels length must match digits rows")

    device = digits.device
    labels = labels.to(device=device, non_blocking=True)
    centers = centers.to(device=device, non_blocking=True)

    correct = torch.tensor(0, dtype=torch.int64, device=device)

    for start in range(0, digits.shape[0], chunk_size):
        chunk = digits[start : start + chunk_size]          # [B, r]
        chunk_labels = labels[start : start + chunk_size]   # [B]

        # [B, C, r] — only chunk_size rows expanded at once
        equal = chunk[:, None, :] == centers[None, :, :]
        # p-adic score: cumulative product along r, summed = shared prefix length
        scores = equal.to(torch.int64).cumprod(dim=-1).sum(dim=-1)  # [B, C]
        predicted = torch.argmax(scores, dim=1)                      # [B]
        correct += (predicted == chunk_labels).sum()

    return float(correct.item() / digits.shape[0])


# ===========================================================================
# 2. DEVICE-AWARE distance kernel
#    All synchronization is explicit; no .item()/.cpu() inside the hot path.
# ===========================================================================

def padic_distance_kernel(
    digits: torch.Tensor,
    pairs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute p-adic valuation scores for a batch of index pairs.

    Returns a 1-D int64 tensor of valuation scores — does NOT call .item()
    or .cpu() so the result stays on-device for downstream loss computation.

    Args:
        digits: [N, r]    int64 Hensel digits (any device)
        pairs:  [M, 2]    int64 row index pairs

    Returns:
        valuations: [M]   int64, higher = closer in p-adic metric
    """
    left  = digits[pairs[:, 0]]   # [M, r]
    right = digits[pairs[:, 1]]   # [M, r]
    equal = left == right          # [M, r] bool
    # cumprod stops at first mismatch — sum = length of shared low-order prefix
    valuations = equal.to(torch.int64).cumprod(dim=-1).sum(dim=-1)  # [M]
    return valuations


def padic_distance_loss(
    digits: torch.Tensor,
    pairs: torch.Tensor,
    target_closer: torch.Tensor,
) -> torch.Tensor:
    """
    Margin-ranking loss in valuation space (stays fully on-device).

    For each triplet (anchor, positive, negative), pushes
    v(anchor, positive) > v(anchor, negative).

    Args:
        digits:         [N, r]
        pairs:          [M, 2]   columns = (query_idx, candidate_idx)
        target_closer:  [M]      bool, True = this pair should be close

    Returns:
        Scalar loss tensor on the same device as digits.
    """
    val = padic_distance_kernel(digits, pairs).float()
    # Convert valuation to distance proxy: larger val = smaller distance
    # Use a simple hinge: push target-close pairs to have val >= margin
    margin = digits.shape[1] * 0.5
    close_vals = val[target_closer]
    far_vals = val[~target_closer]
    loss_close = (
        F.relu(margin - close_vals).mean()
        if close_vals.numel() > 0
        else val.new_tensor(0.0)
    )
    loss_far = (
        F.relu(far_vals - (margin * 0.25)).mean()
        if far_vals.numel() > 0
        else val.new_tensor(0.0)
    )
    return loss_close + loss_far


# ===========================================================================
# 3. LINEAR LAYER TOGGLE: FP32  <->  2-adic valuation-based projection
# ===========================================================================

class FP32Linear(nn.Module):
    """Standard FP32 nn.Linear — baseline."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def mode(self) -> str:
        return "fp32"


class PadicLinear(nn.Module):
    """
    2-adic linear projection layer.

    Pipeline per forward pass:
      1. Quantize float input to int64 Hensel digits  (int64_to_digits)
      2. Compute pairwise valuation scores against learnable centers
      3. Project valuation scores through a small FP32 readout head

    The Hensel quantization step is the only non-differentiable part.
    We use a straight-through estimator (STE) so gradients flow through
    the FP32 residual path during training.

    Args:
        in_features:   input dimension
        out_features:  output dimension
        p:             prime base (2 for 2-adic)
        r:             Hensel precision (8, 16, or 32)
        n_centers:     number of learnable prototype centers
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        p: int = 2,
        r: int = 8,
        n_centers: int = 32,
    ) -> None:
        super().__init__()
        if p < 2:
            raise ValueError("p must be >= 2")
        if not 1 <= r <= 32:
            raise ValueError("r must be in [1, 32]")

        self.in_features  = in_features
        self.out_features = out_features
        self.p            = p
        self.r            = r
        self.n_centers    = n_centers

        # Learnable FP32 projection of valuation scores -> output space
        self.valuation_proj = nn.Linear(n_centers, out_features, bias=True)

        # Learnable centers stored as FP32; quantized on-the-fly
        # Shape: [n_centers, in_features]
        self.centers_fp32 = nn.Parameter(
            torch.randn(n_centers, in_features) * 0.1
        )

        # Residual FP32 bypass (STE path for gradient flow)
        self.residual_proj = nn.Linear(in_features, out_features, bias=False)

        # Scale for quantization: map [-scale, scale] -> [0, p^r)
        self.register_buffer("quant_scale", torch.tensor(float(p ** r) / 2.0))

    @torch.no_grad()
    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """FP32 tensor [*, in_features] -> int64 Hensel digits [*, in_features, r]."""
        modulus = int(self.p ** self.r)
        # Clamp and shift to non-negative integers in [0, modulus)
        x_shifted = (x * (modulus / (2.0 * float(self.quant_scale)))).round().long()
        x_clamped = x_shifted.remainder(modulus)                  # [*, in_features]
        # Expand to Hensel digits: shape [*, in_features, r]
        digits = int64_to_digits(x_clamped, p=self.p, r=self.r)   # [*, in_features, r]
        return digits

    def _valuation_scores(
        self,
        x_digits: torch.Tensor,    # [B, in_features, r]
        c_digits: torch.Tensor,    # [n_centers, in_features, r]
    ) -> torch.Tensor:
        """
        For each (batch, center) pair compute the mean shared-prefix valuation
        across in_features dimensions.

        Returns [B, n_centers] float32 scores.
        """
        B           = x_digits.shape[0]
        n_centers   = c_digits.shape[0]
        in_features = x_digits.shape[1]

        # Broadcast: [B, 1, in_features, r] vs [1, C, in_features, r]
        equal  = x_digits[:, None, :, :] == c_digits[None, :, :, :]   # [B, C, in, r]
        prefix = equal.to(torch.int64).cumprod(dim=-1).sum(dim=-1)     # [B, C, in]
        scores = prefix.float().mean(dim=-1)                           # [B, C]
        return scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features] float32

        Returns:
            [B, out_features] float32
        """
        # Straight-through residual for gradient flow
        residual = self.residual_proj(x)

        # Quantize input and centers to Hensel digits
        x_digits = self._quantize(x)                                    # [B, in, r]
        c_digits = self._quantize(self.centers_fp32)                    # [C, in, r]

        # Valuation scores
        val_scores = self._valuation_scores(x_digits, c_digits)         # [B, C]

        # Project valuation scores to output space
        padic_out = self.valuation_proj(val_scores)                     # [B, out]

        # Combine: padic path + STE residual
        return padic_out + residual

    @property
    def mode(self) -> str:
        return f"2adic_p{self.p}_r{self.r}"


LinearMode = Literal["fp32", "2adic"]


def make_linear(
    in_features: int,
    out_features: int,
    mode: LinearMode = "fp32",
    p: int = 2,
    r: int = 8,
    n_centers: int = 32,
) -> nn.Module:
    """Factory that returns FP32Linear or PadicLinear based on `mode`."""
    if mode == "fp32":
        return FP32Linear(in_features, out_features)
    if mode == "2adic":
        return PadicLinear(in_features, out_features, p=p, r=r, n_centers=n_centers)
    raise ValueError(f"Unknown mode: {mode!r}. Choose 'fp32' or '2adic'.")


# ===========================================================================
# 4. METRIC LOGGER
#    Tracks F1, latency, bit-flip resilience — no external dependencies.
# ===========================================================================

@dataclass
class RunMetrics:
    run_id:          str
    mode:            str
    p:               int
    r:               int
    device:          str
    # --- F1 ---
    tp:              int   = 0
    fp:              int   = 0
    fn:              int   = 0
    # --- latency ---
    latency_samples: list[float] = field(default_factory=list)
    # --- bit-flip resilience ---
    clean_acc:       float = 0.0
    noisy_acc:       float = 0.0
    # --- misc ---
    nearest_center_acc: float = 0.0
    ultrametric_violation_rate: float = 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return (sum(self.latency_samples) / len(self.latency_samples) * 1e3
                if self.latency_samples else float("nan"))

    @property
    def p99_latency_ms(self) -> float:
        if not self.latency_samples:
            return float("nan")
        sorted_lat = sorted(self.latency_samples)
        idx = max(0, int(math.ceil(0.99 * len(sorted_lat))) - 1)
        return sorted_lat[idx] * 1e3

    @property
    def bit_flip_delta(self) -> float:
        """Accuracy drop under bit-flip noise (lower = more resilient)."""
        return self.clean_acc - self.noisy_acc

    def to_dict(self) -> dict:
        return {
            "run_id":                    self.run_id,
            "mode":                      self.mode,
            "p":                         self.p,
            "r":                         self.r,
            "device":                    self.device,
            "f1":                        round(self.f1, 6),
            "precision":                 round(self.precision, 6),
            "recall":                    round(self.recall, 6),
            "mean_latency_ms":           round(self.mean_latency_ms, 4),
            "p99_latency_ms":            round(self.p99_latency_ms, 4),
            "clean_acc":                 round(self.clean_acc, 6),
            "noisy_acc":                 round(self.noisy_acc, 6),
            "bit_flip_delta":            round(self.bit_flip_delta, 6),
            "nearest_center_acc":        round(self.nearest_center_acc, 6),
            "ultrametric_violation_rate": round(self.ultrametric_violation_rate, 6),
        }


class MetricLogger:
    """Accumulates RunMetrics across multiple runs; supports CSV export."""

    def __init__(self) -> None:
        self._runs: list[RunMetrics] = []

    def new_run(self, run_id: str, mode: str, p: int, r: int, device: str) -> RunMetrics:
        m = RunMetrics(run_id=run_id, mode=mode, p=p, r=r, device=device)
        self._runs.append(m)
        return m

    def update_f1(
        self,
        metrics: RunMetrics,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        anomaly_class: int = 1,
    ) -> None:
        """
        Update TP/FP/FN from on-device tensors — no .item() per element.
        Single .item() calls only for the three scalar accumulators.
        """
        pred = predictions.to(dtype=torch.int64)
        tgt  = targets.to(dtype=torch.int64, device=pred.device)
        pos_pred = pred == anomaly_class
        pos_tgt  = tgt  == anomaly_class
        metrics.tp += int((pos_pred & pos_tgt).sum().item())
        metrics.fp += int((pos_pred & ~pos_tgt).sum().item())
        metrics.fn += int((~pos_pred & pos_tgt).sum().item())

    def record_latency(self, metrics: RunMetrics, elapsed_seconds: float) -> None:
        metrics.latency_samples.append(elapsed_seconds)

    def record_resilience(
        self,
        metrics: RunMetrics,
        clean_acc: float,
        noisy_acc: float,
    ) -> None:
        metrics.clean_acc = clean_acc
        metrics.noisy_acc = noisy_acc

    def all_dicts(self) -> list[dict]:
        return [m.to_dict() for m in self._runs]

    def print_table(self) -> None:
        if not self._runs:
            print("No runs logged.")
            return
        header = (
            f"{'run_id':<20} {'mode':<18} {'p':>2} {'r':>2} "
            f"{'F1':>7} {'lat_ms':>8} {'p99_ms':>8} "
            f"{'clean':>7} {'noisy':>7} {'bfd':>7}"
        )
        print(header)
        print("-" * len(header))
        for m in self._runs:
            print(
                f"{m.run_id:<20} {m.mode:<18} {m.p:>2} {m.r:>2} "
                f"{m.f1:>7.4f} {m.mean_latency_ms:>8.3f} {m.p99_latency_ms:>8.3f} "
                f"{m.clean_acc:>7.4f} {m.noisy_acc:>7.4f} {m.bit_flip_delta:>7.4f}"
            )

    def export_csv(self, path: str) -> None:
        import csv
        rows = self.all_dicts()
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Metrics exported to: {path}")


# ===========================================================================
# 5. ExperimentController
# ===========================================================================

@dataclass
class SweepConfig:
    """Full sweep configuration."""
    p_list:            list[int]       = field(default_factory=lambda: [2])
    r_list:            list[int]       = field(default_factory=lambda: [8, 16, 32])
    modes:             list[LinearMode]= field(default_factory=lambda: ["fp32", "2adic"])
    n_samples:         int             = 4096
    n_classes:         int             = 16
    tokens_per_class:  int             = 256
    in_features:       int             = 64
    out_features:      int             = 16
    n_centers:         int             = 32
    batch_size:        int             = 256
    n_latency_batches: int             = 20
    noise_flip_prob:   float           = 0.05
    chunk_size:        int             = 512
    triplets:          int             = 20000
    seed:              int             = 20260504
    device:            str             = "cpu"
    results_dir:       str             = "results"


class ExperimentController:
    """
    Orchestrates a comparative sweep between FP32 and 2-adic linear layers.

    For each (mode, p, r) combination it:
      - Generates a clustered Hensel dataset
      - Runs chunked nearest-center classification
      - Measures inference latency with CUDA synchronization
      - Evaluates bit-flip resilience
      - Logs all metrics via MetricLogger
    """

    def __init__(self, config: SweepConfig) -> None:
        self.cfg    = config
        self.device = torch.device(config.device)
        self.logger = MetricLogger()
        self._rng   = torch.Generator(device=self.device)
        self._rng.manual_seed(config.seed)

    # ------------------------------------------------------------------ #
    # Dataset helpers                                                      #
    # ------------------------------------------------------------------ #

    def _make_dataset(self, p: int, r: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (token_digits, labels, centers) on self.device."""
        from padic_transformer.ultrametric import generate_clustered_hensel_dataset
        bench_cfg = BenchmarkConfig(
            p=p, r=r,
            samples=self.cfg.n_samples,
            classes=self.cfg.n_classes,
            tokens_per_class=self.cfg.tokens_per_class,
            seed=self.cfg.seed,
            triplets=self.cfg.triplets,
        )
        ds = generate_clustered_hensel_dataset(bench_cfg, device=self.device)
        return ds.token_digits, ds.token_labels, ds.center_digits

    def _digits_to_fp32(self, digits: torch.Tensor) -> torch.Tensor:
        """
        Convert int64 Hensel digit tensor [N, r] to float32 [N, r].
        Normalizes digits to [0, 1) for use with FP32 layers.
        """
        p_max = float(digits.max().item() + 1)
        return digits.float() / p_max

    # ------------------------------------------------------------------ #
    # Bit-flip noise                                                       #
    # ------------------------------------------------------------------ #

    def _apply_bit_flip_noise(
        self,
        digits: torch.Tensor,
        p: int,
        flip_prob: float,
    ) -> torch.Tensor:
        """
        Randomly perturb Hensel digits to simulate bit-flip hardware faults.
        Each digit is replaced with a random value in [0, p) with probability
        `flip_prob` — independent of the current digit value.
        """
        mask = torch.bernoulli(
            torch.full(digits.shape, flip_prob, device=digits.device)
        ).bool()
        noise = torch.randint(0, p, digits.shape, dtype=torch.int64, device=digits.device,
                              generator=self._rng)
        return torch.where(mask, noise, digits)

    # ------------------------------------------------------------------ #
    # Latency measurement (CUDA-safe)                                      #
    # ------------------------------------------------------------------ #

    def _time_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> float:
        """
        Time a single forward pass.
        Uses CUDA events when on GPU for wall-clock accuracy;
        falls back to perf_counter on CPU.
        """
        model.eval()
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                _ = model(x)
            end_event.record()
            torch.cuda.synchronize(self.device)
            return start_event.elapsed_time(end_event) / 1e3   # seconds
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            return time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # Ultrametric violation rate (re-used from ultrametric.py)            #
    # ------------------------------------------------------------------ #

    def _violation_rate(self, digits: torch.Tensor) -> float:
        from padic_transformer.ultrametric import ultrametric_violation_rate
        rate, _ = ultrametric_violation_rate(
            digits, triplets=self.cfg.triplets, seed=self.cfg.seed + 99
        )
        return rate

    # ------------------------------------------------------------------ #
    # Single run                                                           #
    # ------------------------------------------------------------------ #

    def _run_one(
        self,
        mode: LinearMode,
        p: int,
        r: int,
    ) -> RunMetrics:
        run_id = f"{mode}_p{p}_r{r}"
        print(f"\n{'='*60}")
        print(f"  Run: {run_id}  |  device: {self.device}")
        print(f"{'='*60}")

        metrics = self.logger.new_run(
            run_id=run_id, mode=mode, p=p, r=r, device=str(self.device)
        )

        # ---- Dataset ----
        digits, labels, centers = self._make_dataset(p, r)
        x_fp32 = self._digits_to_fp32(digits)   # float32 view for FP32 layers

        # ---- Model ----
        model = make_linear(
            in_features  = r,
            out_features = self.cfg.n_classes,
            mode         = mode,
            p            = p,
            r            = r,
            n_centers    = self.cfg.n_centers,
        ).to(self.device)

        # ---- Chunked nearest-center accuracy ----
        print(f"  [1/4] Chunked nearest-center accuracy (chunk={self.cfg.chunk_size})...")
        nca = nearest_center_accuracy_chunked(
            digits, labels, centers, chunk_size=self.cfg.chunk_size
        )
        metrics.nearest_center_acc = nca
        print(f"        -> {nca:.4f}")

        # ---- Ultrametric violation rate ----
        print(f"  [2/4] Ultrametric violation rate ({self.cfg.triplets} triplets)...")
        vr = self._violation_rate(digits)
        metrics.ultrametric_violation_rate = vr
        print(f"        -> {vr:.6f}")

        # ---- Latency (n_latency_batches warm + timed forward passes) ----
        print(f"  [3/4] Latency ({self.cfg.n_latency_batches} batches of {self.cfg.batch_size})...")
        batch_x = x_fp32[: self.cfg.batch_size].to(
            device=self.device, non_blocking=True
        )
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(batch_x)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        # Timed
        for _ in range(self.cfg.n_latency_batches):
            elapsed = self._time_forward(model, batch_x)
            self.logger.record_latency(metrics, elapsed)
        print(f"        -> mean {metrics.mean_latency_ms:.3f} ms  p99 {metrics.p99_latency_ms:.3f} ms")

        # ---- Bit-flip resilience ----
        print(f"  [4/4] Bit-flip resilience (flip_prob={self.cfg.noise_flip_prob})...")
        clean_acc = nearest_center_accuracy_chunked(
            digits, labels, centers, chunk_size=self.cfg.chunk_size
        )
        noisy_digits = self._apply_bit_flip_noise(digits, p, self.cfg.noise_flip_prob)
        noisy_acc = nearest_center_accuracy_chunked(
            noisy_digits, labels, centers, chunk_size=self.cfg.chunk_size
        )
        self.logger.record_resilience(metrics, clean_acc, noisy_acc)
        print(f"        -> clean={clean_acc:.4f}  noisy={noisy_acc:.4f}  delta={metrics.bit_flip_delta:.4f}")

        # ---- F1 (nearest-center predictions as classification) ----
        # Use chunked argmax predictions vs ground-truth labels
        with torch.no_grad():
            scores_list = []
            for start in range(0, digits.shape[0], self.cfg.chunk_size):
                chunk = digits[start : start + self.cfg.chunk_size]
                equal = chunk[:, None, :] == centers[None, :, :]
                scores = equal.to(torch.int64).cumprod(dim=-1).sum(dim=-1)
                scores_list.append(scores)
            all_scores = torch.cat(scores_list, dim=0)
            predictions = torch.argmax(all_scores, dim=1)

        self.logger.update_f1(metrics, predictions, labels, anomaly_class=0)
        print(f"        F1={metrics.f1:.4f}  P={metrics.precision:.4f}  R={metrics.recall:.4f}")

        return metrics

    # ------------------------------------------------------------------ #
    # Full sweep                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> list[dict]:
        """Execute all (mode, p, r) combinations and return metric dicts."""
        results = []
        for mode in self.cfg.modes:
            for p in self.cfg.p_list:
                for r in self.cfg.r_list:
                    # 2-adic only makes sense for p=2
                    if mode == "2adic" and p != 2:
                        continue
                    m = self._run_one(mode, p, r)
                    results.append(m.to_dict())

        print("\n")
        self.logger.print_table()

        import os
        os.makedirs(self.cfg.results_dir, exist_ok=True)
        csv_path = f"{self.cfg.results_dir}/experiment_results.csv"
        self.logger.export_csv(csv_path)
        return results


# ===========================================================================
# 6. GPU SMOKE TEST
#    Fast pre-flight validation before committing to the full sweep.
# ===========================================================================

def gpu_smoke_test(device_str: str = "cpu") -> None:
    """
    Verifies all kernels end-to-end on a small synthetic dataset.
    Should complete in < 10 seconds on CPU, < 2 seconds on GPU.

    Checks:
      - PadicLinear and FP32Linear forward pass shapes
      - chunked nearest_center_accuracy matches eager result
      - padic_distance_kernel output range and dtype
      - MetricLogger F1 computation
      - CUDA non-blocking transfer correctness
      - Bit-flip noise preserves digit range
    """
    print(f"\n{'#'*60}")
    print(f"  GPU SMOKE TEST  |  device={device_str}")
    print(f"{'#'*60}\n")

    device = torch.device(device_str)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    SEED   = 42
    N      = 256
    C      = 8
    R      = 8
    P      = 2
    IN_F   = R
    OUT_F  = C
    CHUNK  = 64

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)

    # Synthetic Hensel digits [N, R] with digits in [0, P)
    digits  = torch.randint(0, P, (N, R), dtype=torch.int64,
                            device=device, generator=gen)
    labels  = torch.randint(0, C, (N,),  dtype=torch.int64,
                            device=device, generator=gen)
    centers = torch.randint(0, P, (C, R), dtype=torch.int64,
                            device=device, generator=gen)

    # --- Test 1: chunked vs eager nearest_center_accuracy ---
    print("  [1] Chunked vs eager nearest_center_accuracy...")
    chunked_acc = nearest_center_accuracy_chunked(digits, labels, centers, chunk_size=CHUNK)

    equal_eager  = digits[:, None, :] == centers[None, :, :]
    scores_eager = equal_eager.to(torch.int64).cumprod(dim=-1).sum(dim=-1)
    pred_eager   = torch.argmax(scores_eager, dim=1)
    eager_acc    = float((pred_eager == labels).float().mean().item())

    assert abs(chunked_acc - eager_acc) < 1e-6, (
        f"Chunked/eager mismatch: {chunked_acc:.6f} != {eager_acc:.6f}"
    )
    print(f"     PASS  acc={chunked_acc:.4f}")

    # --- Test 2: padic_distance_kernel dtype and range ---
    print("  [2] padic_distance_kernel dtype and range...")
    M = 512
    pairs = torch.randint(0, N, (M, 2), dtype=torch.int64, device=device)
    val = padic_distance_kernel(digits, pairs)
    assert val.dtype == torch.int64, f"Expected int64, got {val.dtype}"
    assert int(val.min().item()) >= 0
    assert int(val.max().item()) <= R
    print(f"     PASS  val range=[{int(val.min().item())}, {int(val.max().item())}]  dtype={val.dtype}")

    # --- Test 3: FP32Linear forward shape ---
    print("  [3] FP32Linear forward shape...")
    fp32_model = FP32Linear(IN_F, OUT_F).to(device)
    x_fp32 = digits.float() / P
    out_fp32 = fp32_model(x_fp32)
    assert out_fp32.shape == (N, OUT_F), f"Shape mismatch: {out_fp32.shape}"
    assert not torch.isnan(out_fp32).any(), "NaN in FP32Linear output"
    print(f"     PASS  output shape={tuple(out_fp32.shape)}")

    # --- Test 4: PadicLinear forward shape ---
    print("  [4] PadicLinear forward shape and no NaN...")
    padic_model = PadicLinear(IN_F, OUT_F, p=P, r=R, n_centers=C).to(device)
    out_padic = padic_model(x_fp32)
    assert out_padic.shape == (N, OUT_F), f"Shape mismatch: {out_padic.shape}"
    assert not torch.isnan(out_padic).any(), "NaN in PadicLinear output"
    print(f"     PASS  output shape={tuple(out_padic.shape)}")

    # --- Test 5: Non-blocking CUDA transfer (no-op on CPU) ---
    print("  [5] Non-blocking device transfer...")
    cpu_digits = torch.randint(0, P, (N, R), dtype=torch.int64)
    transferred = cpu_digits.to(device=device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    assert transferred.device.type == device.type
    print(f"     PASS  tensor on {transferred.device}")

    # --- Test 6: Bit-flip noise preserves digit range ---
    print("  [6] Bit-flip noise preserves digit range...")
    ctrl = ExperimentController(SweepConfig(device=device_str, seed=SEED))
    noisy = ctrl._apply_bit_flip_noise(digits, P, flip_prob=0.1)
    assert int(noisy.min().item()) >= 0
    assert int(noisy.max().item()) < P
    print(f"     PASS  noisy range=[{int(noisy.min().item())}, {int(noisy.max().item())}]")

    # --- Test 7: MetricLogger F1 ---
    print("  [7] MetricLogger F1 computation...")
    logger = MetricLogger()
    m = logger.new_run("smoke", "fp32", 2, 8, device_str)
    # Synthetic perfect predictions for class 0
    perfect_preds   = torch.zeros(N, dtype=torch.int64, device=device)
    perfect_targets = torch.zeros(N, dtype=torch.int64, device=device)
    logger.update_f1(m, perfect_preds, perfect_targets, anomaly_class=0)
    assert abs(m.f1 - 1.0) < 1e-6, f"Expected F1=1.0 for perfect preds, got {m.f1}"
    print(f"     PASS  F1={m.f1:.4f} (perfect predictions)")

    # --- Test 8: CUDA event timing (skipped on CPU) ---
    if device.type == "cuda":
        print("  [8] CUDA event timing...")
        ctrl2 = ExperimentController(SweepConfig(device=device_str))
        elapsed = ctrl2._time_forward(fp32_model, x_fp32)
        assert elapsed > 0, "Elapsed time must be positive"
        print(f"     PASS  elapsed={elapsed*1e3:.3f} ms")
    else:
        print("  [8] CUDA event timing... SKIPPED (CPU device)")

    print(f"\n{'#'*60}")
    print(f"  ALL SMOKE TESTS PASSED  |  device={device_str}")
    print(f"{'#'*60}\n")


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run GPU smoke test and exit")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Compute device (default: cpu)")
    parser.add_argument("--p-list", nargs="+", type=int, default=[2],
                        help="List of primes (default: 2)")
    parser.add_argument("--r-list", nargs="+", type=int, default=[8, 16, 32],
                        help="List of precisions (default: 8 16 32)")
    parser.add_argument("--modes", nargs="+", default=["fp32", "2adic"],
                        choices=["fp32", "2adic"],
                        help="Layer modes to compare")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--classes", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke_test:
        gpu_smoke_test(device_str=args.device)
        return

    cfg = SweepConfig(
        p_list           = args.p_list,
        r_list           = args.r_list,
        modes            = args.modes,
        n_samples        = args.samples,
        n_classes        = args.classes,
        device           = args.device,
        batch_size       = args.batch_size,
        chunk_size       = args.chunk_size,
        seed             = args.seed,
        results_dir      = args.results_dir,
    )
    ctrl = ExperimentController(cfg)
    ctrl.run()


if __name__ == "__main__":
    main()

"""Synthetic p-adic tree generation and ultrametric validation in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import BenchmarkConfig
from .hensel import shared_prefix_valuation


@dataclass(frozen=True)
class HenselDataset:
    token_digits: torch.Tensor
    token_labels: torch.Tensor
    center_digits: torch.Tensor
    cluster_depth: int


def generate_clustered_hensel_dataset(
    config: BenchmarkConfig,
    *,
    device: torch.device | str = "cpu",
) -> HenselDataset:
    """Generate clustered Hensel codes from a p-adic tree.

    Each class owns a center. Tokens in the class share the center's low-order
    Hensel prefix and randomize the remaining digits. This gives a direct
    synthetic ultrametric latent structure.
    """
    config.validate()
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(config.seed + config.p * 1000 + config.r)
    cluster_depth = max(1, config.r // 2)

    centers = torch.randint(
        0,
        config.p,
        (config.classes, config.r),
        dtype=torch.int64,
        device=resolved_device,
        generator=generator,
    )
    token_count = config.classes * config.tokens_per_class
    digits = torch.randint(
        0,
        config.p,
        (token_count, config.r),
        dtype=torch.int64,
        device=resolved_device,
        generator=generator,
    )
    labels = torch.arange(config.classes, dtype=torch.int64, device=resolved_device).repeat_interleave(
        config.tokens_per_class
    )

    for class_id in range(config.classes):
        start = class_id * config.tokens_per_class
        stop = start + config.tokens_per_class
        digits[start:stop, :cluster_depth] = centers[class_id, :cluster_depth]

    if config.samples < token_count:
        selected = torch.randperm(token_count, device=resolved_device, generator=generator)[
            : config.samples
        ]
        digits = digits[selected]
        labels = labels[selected]
    elif config.samples > token_count:
        selected = torch.randint(
            0,
            token_count,
            (config.samples - token_count,),
            dtype=torch.int64,
            device=resolved_device,
            generator=generator,
        )
        digits = torch.cat([digits, digits[selected]], dim=0)
        labels = torch.cat([labels, labels[selected]], dim=0)

    return HenselDataset(
        token_digits=digits,
        token_labels=labels,
        center_digits=centers,
        cluster_depth=cluster_depth,
    )


def ultrametric_violation_rate(
    digits: torch.Tensor,
    *,
    triplets: int,
    seed: int,
) -> tuple[float, int]:
    """Estimate ultrametric violations over sampled triplets."""
    arr = torch.as_tensor(digits)
    if arr.ndim != 2:
        raise ValueError("digits must have shape [items, r]")
    if arr.shape[0] < 3:
        raise ValueError("at least three rows are required")
    if triplets < 1:
        raise ValueError("triplets must be positive")

    generator = torch.Generator(device=arr.device)
    generator.manual_seed(seed)
    ids = torch.randint(
        0,
        arr.shape[0],
        (triplets, 3),
        dtype=torch.int64,
        device=arr.device,
        generator=generator,
    )
    x = arr[ids[:, 0]]
    y = arr[ids[:, 1]]
    z = arr[ids[:, 2]]

    v_xy = shared_prefix_valuation(x, y)
    v_yz = shared_prefix_valuation(y, z)
    v_xz = shared_prefix_valuation(x, z)
    violations = v_xz < torch.minimum(v_xy, v_yz)
    count = int(violations.sum().item())
    return count / float(triplets), count


def nearest_center_accuracy(digits: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor) -> float:
    """Classify tokens by nearest p-adic center and report accuracy."""
    arr = torch.as_tensor(digits)
    labs = torch.as_tensor(labels, device=arr.device)
    ctr = torch.as_tensor(centers, device=arr.device)
    if arr.ndim != 2 or ctr.ndim != 2:
        raise ValueError("digits and centers must both have shape [items, r]")
    if arr.shape[1] != ctr.shape[1]:
        raise ValueError("digits and centers must share the same precision")
    if labs.shape[0] != arr.shape[0]:
        raise ValueError("labels length must match digits rows")

    equal = arr[:, None, :] == ctr[None, :, :]
    scores = equal.to(torch.int64).cumprod(dim=-1).sum(dim=-1)
    best = torch.argmax(scores, dim=1)
    return float((best == labs).to(torch.float32).mean().item())

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
    center_labels: torch.Tensor
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
    class_counts = [config.tokens_per_class for _ in range(config.classes)]
    base_token_count = config.classes * config.tokens_per_class
    if config.samples > base_token_count:
        extra = config.samples - base_token_count
        for class_id in range(config.classes):
            class_counts[class_id] += extra // config.classes
        for class_id in range(extra % config.classes):
            class_counts[class_id] += 1

    token_count = sum(class_counts)
    digits = torch.randint(
        0,
        config.p,
        (token_count, config.r),
        dtype=torch.int64,
        device=resolved_device,
        generator=generator,
    )
    labels = torch.repeat_interleave(
        torch.arange(config.classes, dtype=torch.int64, device=resolved_device),
        torch.tensor(class_counts, dtype=torch.int64, device=resolved_device),
    )

    start = 0
    for class_id in range(config.classes):
        stop = start + class_counts[class_id]
        digits[start:stop, :cluster_depth] = centers[class_id, :cluster_depth]
        start = stop

    if config.samples < token_count:
        selected = torch.randperm(token_count, device=resolved_device, generator=generator)[
            : config.samples
        ]
        digits = digits[selected]
        labels = labels[selected]

    return HenselDataset(
        token_digits=digits,
        token_labels=labels,
        center_digits=centers,
        center_labels=torch.arange(config.classes, dtype=torch.int64, device=resolved_device),
        cluster_depth=cluster_depth,
    )


def map_raw_indices_excluding_pair(
    raw: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Map raw indices into a range with two sorted excluded positions removed."""
    raw_values = torch.as_tensor(raw, dtype=torch.int64)
    lower_values = torch.as_tensor(lower, dtype=torch.int64, device=raw_values.device)
    upper_values = torch.as_tensor(upper, dtype=torch.int64, device=raw_values.device)
    if bool(torch.any(lower_values >= upper_values).item()):
        raise ValueError("lower must be strictly less than upper")

    shifted = raw_values + (raw_values >= lower_values).to(torch.int64)
    return shifted + (shifted >= upper_values).to(torch.int64)


def sample_distinct_triplet_indices(
    population: int,
    triplets: int,
    *,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample triplet indices with x, y, and z distinct in every row."""
    if population < 3:
        raise ValueError("population must be at least 3")
    if triplets < 1:
        raise ValueError("triplets must be positive")

    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(seed)

    x = torch.randint(
        0,
        population,
        (triplets,),
        dtype=torch.int64,
        device=resolved_device,
        generator=generator,
    )
    y_raw = torch.randint(
        0,
        population - 1,
        (triplets,),
        dtype=torch.int64,
        device=resolved_device,
        generator=generator,
    )
    y = y_raw + (y_raw >= x).to(torch.int64)

    lower = torch.minimum(x, y)
    upper = torch.maximum(x, y)
    z_raw = torch.randint(
        0,
        population - 2,
        (triplets,),
        dtype=torch.int64,
        device=resolved_device,
        generator=generator,
    )
    z = map_raw_indices_excluding_pair(z_raw, lower, upper)
    return torch.stack([x, y, z], dim=1)


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

    ids = sample_distinct_triplet_indices(
        arr.shape[0],
        triplets,
        seed=seed,
        device=arr.device,
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


def nearest_center_accuracy(
    digits: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    center_labels: torch.Tensor | None = None,
) -> float:
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

    if center_labels is None:
        center_label_values = torch.arange(ctr.shape[0], dtype=torch.int64, device=arr.device)
    else:
        center_label_values = torch.as_tensor(center_labels, dtype=torch.int64, device=arr.device)
        if center_label_values.ndim != 1 or center_label_values.shape[0] != ctr.shape[0]:
            raise ValueError("center_labels must have one entry per center row")

    equal = arr[:, None, :] == ctr[None, :, :]
    scores = equal.to(torch.int64).cumprod(dim=-1).sum(dim=-1)
    best = center_label_values[torch.argmax(scores, dim=1)]
    return float((best == labs).to(torch.float32).mean().item())

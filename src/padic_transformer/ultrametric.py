"""Synthetic p-adic tree generation and ultrametric validation in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import warnings

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


def derive_seed(base_seed: int, stream: str) -> int:
    """Derive a deterministic 63-bit child seed for independent dataset streams."""
    payload = f"{base_seed}:{stream}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


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
    if config.samples < config.classes:
        raise ValueError(
            f"samples ({config.samples}) must be at least classes ({config.classes}) "
            "to guarantee class coverage"
        )
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
    class_token_bank = torch.randint(
        0,
        config.p,
        (config.classes, config.tokens_per_class, config.r),
        dtype=torch.int64,
        device=resolved_device,
        generator=generator,
    )
    for class_id in range(config.classes):
        class_token_bank[class_id, :, :cluster_depth] = centers[class_id, :cluster_depth]

    labels = torch.empty(config.samples, dtype=torch.int64, device=resolved_device)
    max_dwell = max(2, min(cluster_depth + 1, 8))
    current_label = int(
        torch.randint(
            0,
            config.classes,
            (1,),
            dtype=torch.int64,
            device=resolved_device,
            generator=generator,
        ).item()
    )

    cursor = 0
    while cursor < config.samples:
        dwell = int(
            torch.randint(
                1,
                max_dwell + 1,
                (1,),
                dtype=torch.int64,
                device=resolved_device,
                generator=generator,
            ).item()
        )
        stop = min(config.samples, cursor + dwell)
        labels[cursor:stop] = current_label
        cursor = stop
        if cursor >= config.samples:
            break
        raw_next = int(
            torch.randint(
                0,
                config.classes - 1,
                (1,),
                dtype=torch.int64,
                device=resolved_device,
                generator=generator,
            ).item()
        )
        current_label = raw_next + (1 if raw_next >= current_label else 0)

    class_counts = torch.bincount(labels, minlength=config.classes)
    missing_classes = (class_counts == 0).nonzero(as_tuple=True)[0]
    if missing_classes.numel() > 0:
        for missing_class in missing_classes.tolist():
            surplus_classes = (class_counts > 1).nonzero(as_tuple=True)[0]
            if surplus_classes.numel() == 0:
                raise RuntimeError(
                    "unable to guarantee class coverage without removing another class"
                )
            surplus_pick = int(
                torch.randint(
                    0,
                    surplus_classes.numel(),
                    (1,),
                    device=resolved_device,
                    generator=generator,
                ).item()
            )
            source_class = int(surplus_classes[surplus_pick].item())
            source_positions = (labels == source_class).nonzero(as_tuple=True)[0]
            position_pick = int(
                torch.randint(
                    0,
                    source_positions.numel(),
                    (1,),
                    device=resolved_device,
                    generator=generator,
                ).item()
            )
            labels[source_positions[position_pick]] = missing_class
            class_counts[source_class] -= 1
            class_counts[missing_class] += 1

    digits = torch.empty((config.samples, config.r), dtype=torch.int64, device=resolved_device)
    for class_id in range(config.classes):
        mask = labels == class_id
        count = int(mask.sum().item())
        if count == 0:
            continue
        token_ids = torch.randint(
            0,
            config.tokens_per_class,
            (count,),
            dtype=torch.int64,
            device=resolved_device,
            generator=generator,
        )
        digits[mask] = class_token_bank[class_id, token_ids]

    min_required = max(1, config.tokens_per_class // 8)
    class_counts = torch.bincount(labels, minlength=config.classes)
    sparse_classes = (class_counts < min_required).sum().item()
    if sparse_classes > 0:
        warnings.warn(
            f"generate_clustered_hensel_dataset: {sparse_classes}/{config.classes} classes "
            f"have fewer than {min_required} tokens (min count = {int(class_counts.min().item())}). "
            "Increase `samples` or decrease `classes` for better class coverage.",
            RuntimeWarning,
            stacklevel=2,
        )

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

    # y is sampled by skipping x, so lower < upper is guaranteed here.
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
    chunk_size: int = 8192,
) -> float:
    """Classify tokens by nearest p-adic center and report accuracy.

    The computation is chunked to avoid allocating a full [N, C, r] tensor for
    large benchmark sweeps.
    """
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
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    correct = 0
    for start in range(0, arr.shape[0], chunk_size):
        chunk = arr[start : start + chunk_size]
        equal = chunk[:, None, :] == ctr[None, :, :]
        scores = equal.to(torch.int64).cumprod(dim=-1).sum(dim=-1)
        best = center_label_values[torch.argmax(scores, dim=1)]
        correct += int((best == labs[start : start + chunk_size]).sum().item())
    return correct / float(max(1, arr.shape[0]))

"""Shared evaluation metrics for anomaly detection experiments."""

from __future__ import annotations

import torch


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute exact AUROC from scores and binary labels using average ranks."""
    flat_scores = torch.as_tensor(scores, dtype=torch.float64).flatten()
    flat_labels = torch.as_tensor(labels, dtype=torch.int64).flatten()
    if flat_scores.numel() != flat_labels.numel():
        raise ValueError("scores and labels must have the same number of elements")

    pos_mask = flat_labels == 1
    neg_mask = flat_labels == 0
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = torch.argsort(flat_scores, stable=True)
    sorted_scores = flat_scores[order]
    ranks = torch.empty_like(sorted_scores, dtype=torch.float64)

    start = 0
    total = sorted_scores.numel()
    while start < total:
        end = start + 1
        while end < total and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[start:end] = avg_rank
        start = end

    full_ranks = torch.empty_like(ranks)
    full_ranks[order] = ranks
    pos_rank_sum = float(full_ranks[pos_mask].sum().item())
    u_stat = pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)
    return float(u_stat / (n_pos * n_neg))

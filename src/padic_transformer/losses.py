"""Loss functions for p-adic anomaly detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PadicContrastiveLoss(nn.Module):
    def __init__(
        self,
        p: int,
        margin_pos: float = 0.1,
        margin_neg: float = 0.5,
        max_pairs: int = 4096,
    ) -> None:
        super().__init__()
        if p < 2:
            raise ValueError("p must be >= 2")
        if margin_pos >= margin_neg:
            raise ValueError("margin_pos must be < margin_neg")
        self.p = p
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.max_pairs = max_pairs

    def forward(self, representations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if representations.ndim != 2:
            raise ValueError(
                f"representations must have shape [batch, dim], got {tuple(representations.shape)}"
            )
        batch = representations.shape[0]
        rep = F.normalize(representations, dim=-1)

        idx = torch.arange(batch, device=representations.device)
        pairs_i, pairs_j = torch.meshgrid(idx, idx, indexing="ij")
        mask_upper = pairs_i < pairs_j
        pairs_i = pairs_i[mask_upper]
        pairs_j = pairs_j[mask_upper]

        n_pairs = pairs_i.shape[0]
        if n_pairs == 0:
            return representations.new_tensor(0.0, dtype=torch.float32)
        if n_pairs > self.max_pairs:
            perm = torch.randperm(n_pairs, device=representations.device)[: self.max_pairs]
            pairs_i = pairs_i[perm]
            pairs_j = pairs_j[perm]

        a = rep[pairs_i]
        b = rep[pairs_j]
        distance = 1.0 - (a * b).sum(dim=-1)

        same = (labels[pairs_i] == labels[pairs_j]).float()
        diff = 1.0 - same

        pos_loss = same * F.relu(distance - self.margin_pos) ** 2
        neg_loss = diff * F.relu(self.margin_neg - distance) ** 2
        return (pos_loss + neg_loss).mean()


class AnomalyLoss(nn.Module):
    def __init__(
        self,
        p: int,
        alpha: float = 0.5,
        pos_weight: float | None = None,
        margin_pos: float = 0.1,
        margin_neg: float = 0.5,
        max_pairs: int = 4096,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor([pos_weight]))
        else:
            self._pos_weight = None
        self.contrastive = PadicContrastiveLoss(
            p=p,
            margin_pos=margin_pos,
            margin_neg=margin_neg,
            max_pairs=max_pairs,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        representations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self._pos_weight
            )
        else:
            bce_loss = self.bce(logits, labels)
        if self.alpha == 0.0:
            contrastive_loss = logits.new_tensor(0.0)
        else:
            contrastive_loss = self.contrastive(representations, labels)
        total = bce_loss + self.alpha * contrastive_loss
        return total, bce_loss, contrastive_loss

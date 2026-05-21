"""P-adic transformer encoder with binary anomaly detection head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_fixes import HenselEmbeddingWithPosition


class HenselEmbedding(HenselEmbeddingWithPosition):
    """Backward-compatible embedding with optional positional encoding."""

    def __init__(
        self,
        p: int,
        r: int,
        d_model: int,
        max_seq_len: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(p=p, r=r, d_model=d_model, max_seq_len=max_seq_len, dropout=dropout)


class PadicTransformerEncoder(nn.Module):
    """Multi-layer transformer encoder tailored for p-adic syscall sequences."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class AnomalyHead(nn.Module):
    """Mean-pool + two-layer MLP -> single binary logit."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.token_score = nn.Linear(d_model, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def pool_hidden(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).to(hidden.dtype)
            mean_pool = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            scores = self.token_score(hidden).masked_fill(padding_mask.unsqueeze(-1), -1e9)
        else:
            mean_pool = hidden.mean(dim=1)
            scores = self.token_score(hidden)

        weights = torch.softmax(scores, dim=1)
        attn_pool = (weights * hidden).sum(dim=1)
        return torch.cat([mean_pool, attn_pool], dim=-1)

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.pool_hidden(hidden, padding_mask=padding_mask)
        return self.net(pooled).squeeze(-1)


class PadicAnomalyDetector(nn.Module):
    """End-to-end p-adic anomaly detector: embedding -> encoder -> binary logit."""

    def __init__(
        self,
        p: int,
        r: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        head_hidden: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.p = p
        self.r = r
        self.d_model = d_model

        self.embedding = HenselEmbedding(p=p, r=r, d_model=d_model, max_seq_len=max_seq_len, dropout=dropout)
        self.encoder = PadicTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.head = AnomalyHead(d_model=d_model, hidden_dim=head_hidden, dropout=dropout)

    def encode(
        self,
        digits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(digits)
        return self.encoder(x, src_key_padding_mask=padding_mask)

    def forward_with_features(
        self,
        digits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(digits, padding_mask=padding_mask)
        pooled = self.head.pool_hidden(hidden, padding_mask=padding_mask)
        logits = self.head.net(pooled).squeeze(-1)
        return logits, pooled

    def forward(
        self,
        digits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _ = self.forward_with_features(digits, padding_mask=padding_mask)
        return logits

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def parameter_summary(self) -> str:
        total = self.count_parameters()
        lines = [
            f"PadicAnomalyDetector  p={self.p}  r={self.r}  d_model={self.d_model}",
            f"  HenselEmbedding : {sum(param.numel() for param in self.embedding.parameters()):>10,}",
            f"  TransformerEncoder: {sum(param.numel() for param in self.encoder.parameters()):>10,}",
            f"  AnomalyHead     : {sum(param.numel() for param in self.head.parameters()):>10,}",
            f"  -- Total        : {total:>10,}",
        ]
        return "\n".join(lines)

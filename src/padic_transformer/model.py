"""P-adic transformer encoder with binary anomaly detection head."""

from __future__ import annotations

import torch
import torch.nn as nn


class HenselEmbedding(nn.Module):
    """Embed a [batch, seq, r] digit tensor into [batch, seq, d_model]."""

    def __init__(self, p: int, r: int, d_model: int) -> None:
        super().__init__()
        if p < 2:
            raise ValueError("p must be >= 2")
        if r < 1:
            raise ValueError("r must be >= 1")
        if d_model < 1:
            raise ValueError("d_model must be >= 1")
        self.p = p
        self.r = r
        self.d_model = d_model
        self.digit_embeds = nn.ModuleList([nn.Embedding(p, d_model) for _ in range(r)])
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for embed in self.digit_embeds:
            nn.init.normal_(embed.weight, mean=0.0, std=self.d_model ** -0.5)

    def forward(self, digits: torch.Tensor) -> torch.Tensor:
        if digits.ndim != 3 or digits.shape[-1] != self.r:
            raise ValueError(
                f"digits must be shape [batch, seq, {self.r}], got {tuple(digits.shape)}"
            )
        out = torch.zeros(
            digits.shape[0],
            digits.shape[1],
            self.d_model,
            dtype=torch.get_default_dtype(),
            device=digits.device,
        )
        for idx, embed in enumerate(self.digit_embeds):
            out = out + embed(digits[..., idx])
        return out


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
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
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
            mask = (~padding_mask).unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return hidden.mean(dim=1)

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
    ) -> None:
        super().__init__()
        self.p = p
        self.r = r
        self.d_model = d_model

        self.embedding = HenselEmbedding(p=p, r=r, d_model=d_model)
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

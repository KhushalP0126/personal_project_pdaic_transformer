"""Soft p-adic attention layers for anomaly detection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPadicValuation(nn.Module):
    """Differentiable approximation of shared low-order Hensel prefix length."""

    def __init__(
        self,
        p: int,
        r: int,
        d_digit: int = 16,
        temperature: float = 4.0,
        learn_temp: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if p < 2:
            raise ValueError("p must be >= 2")
        if r < 1:
            raise ValueError("r must be >= 1")
        if d_digit < 1:
            raise ValueError("d_digit must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.p = p
        self.r = r
        self.d_digit = d_digit
        self.eps = eps

        self.digit_embeddings = nn.ModuleList([nn.Embedding(p, d_digit) for _ in range(r)])
        if learn_temp:
            self.log_temperature = nn.Parameter(torch.full((r,), math.log(temperature)))
        else:
            self.register_buffer("log_temperature", torch.full((r,), math.log(temperature)))
        self.prefix_weights = nn.Parameter(torch.ones(r))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for emb in self.digit_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=self.d_digit ** -0.5)

    def _soft_match_at_position(self, digits_a: torch.Tensor, digits_b: torch.Tensor, position: int) -> torch.Tensor:
        emb = self.digit_embeddings[position]
        ea = emb(digits_a)
        eb = emb(digits_b)
        sq_dist = ((ea - eb) ** 2).sum(dim=-1)
        tau = self.log_temperature[position].exp().clamp(min=0.01, max=50.0)
        return torch.exp(-tau * sq_dist)

    def forward(self, digits_a: torch.Tensor, digits_b: torch.Tensor) -> torch.Tensor:
        if digits_a.shape != digits_b.shape:
            raise ValueError("digits_a and digits_b must have the same shape")
        if digits_a.shape[-1] != self.r:
            raise ValueError(f"Last dimension must be r={self.r}")

        log_matches = []
        for pos in range(self.r):
            match = self._soft_match_at_position(digits_a[..., pos], digits_b[..., pos], pos)
            log_matches.append(torch.log(match.clamp_min(self.eps)))
        log_match_tensor = torch.stack(log_matches, dim=-1)
        log_prefix = torch.cumsum(log_match_tensor, dim=-1)
        prefix = torch.exp(log_prefix)
        weights = F.softplus(self.prefix_weights)
        soft_val = (prefix * weights).sum(dim=-1)
        return soft_val / (weights.sum() + self.eps)


class PadicAttentionHead(nn.Module):
    """Single attention head using soft p-adic valuation as logits."""

    def __init__(
        self,
        p: int,
        r: int,
        d_model: int,
        d_head: int,
        d_digit: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.valuation = SoftPadicValuation(p=p, r=r, d_digit=d_digit)
        self.value_proj = nn.Linear(d_model, d_head)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        digits: torch.Tensor,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if digits.ndim != 3 or x.ndim != 3:
            raise ValueError("digits and x must be [batch, seq, ...]")
        if digits.shape[:2] != x.shape[:2]:
            raise ValueError("digits and x must have matching batch/seq dimensions")

        batch, seq, _ = digits.shape
        flat_a = (
            digits.unsqueeze(2)
            .expand(-1, -1, seq, -1)
            .contiguous()
            .reshape(batch * seq, seq, self.valuation.r)
        )
        flat_b = (
            digits.unsqueeze(1)
            .expand(-1, seq, -1, -1)
            .contiguous()
            .reshape(batch * seq, seq, self.valuation.r)
        )
        logits = self.valuation(flat_a, flat_b).reshape(batch, seq, seq).to(dtype=x.dtype)

        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask.unsqueeze(1), torch.finfo(logits.dtype).min)

        weights = torch.softmax(logits, dim=-1)
        values = self.value_proj(x)
        out = torch.bmm(weights, values)
        out = self.dropout(out)
        if key_padding_mask is not None:
            out = out * (~key_padding_mask).unsqueeze(-1).to(out.dtype)
        return out, weights


class PadicMultiHeadAttention(nn.Module):
    def __init__(
        self,
        p: int,
        r: int,
        d_model: int,
        n_heads: int,
        d_digit: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.heads = nn.ModuleList(
            [
                PadicAttentionHead(p=p, r=r, d_model=d_model, d_head=d_model // n_heads, d_digit=d_digit, dropout=dropout)
                for _ in range(n_heads)
            ]
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        digits: torch.Tensor,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        outputs = []
        weights = []
        for head in self.heads:
            out, attn = head(digits, x, key_padding_mask=key_padding_mask)
            outputs.append(out)
            weights.append(attn)
        return self.out_proj(torch.cat(outputs, dim=-1)), weights


class PadicTransformerLayer(nn.Module):
    def __init__(
        self,
        p: int,
        r: int,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        d_digit: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = PadicMultiHeadAttention(p=p, r=r, d_model=d_model, n_heads=n_heads, d_digit=d_digit, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        digits: torch.Tensor,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attn_out, weights = self.self_attn(digits, self.norm1(x), key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)
        x = x + self.dropout(self.ffn(x))
        return x, weights


class PadicAttentionEncoder(nn.Module):
    def __init__(
        self,
        p: int,
        r: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        d_digit: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PadicTransformerLayer(
                    p=p,
                    r=r,
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    d_digit=d_digit,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        digits: torch.Tensor,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        all_weights = []
        for layer in self.layers:
            x, weights = layer(digits, x, src_key_padding_mask=src_key_padding_mask)
            all_weights.append(weights)
        return x, all_weights


class HenselEmbedding(nn.Module):
    def __init__(self, p: int, r: int, d_model: int) -> None:
        super().__init__()
        self.p = p
        self.r = r
        self.d_model = d_model
        self.digit_embeds = nn.ModuleList([nn.Embedding(p, d_model) for _ in range(r)])
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for embed in self.digit_embeds:
            nn.init.normal_(embed.weight, mean=0.0, std=self.d_model ** -0.5)

    def forward(self, digits: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(digits.shape[0], digits.shape[1], self.d_model, dtype=torch.get_default_dtype(), device=digits.device)
        for idx, embed in enumerate(self.digit_embeds):
            out = out + embed(digits[..., idx])
        return out


class AnomalyHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def pool_hidden(self, hidden: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return hidden.mean(dim=1)

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self.pool_hidden(hidden, padding_mask=padding_mask)
        return self.net(pooled).squeeze(-1)


class PadicAttentionAnomalyDetector(nn.Module):
    def __init__(
        self,
        p: int,
        r: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        head_hidden: int = 128,
        d_digit: int = 16,
        dropout: float = 0.1,
        ) -> None:
        super().__init__()
        self.p = p
        self.r = r
        self.d_model = d_model
        self.embedding = HenselEmbedding(p=p, r=r, d_model=d_model)
        self.encoder = PadicAttentionEncoder(
            p=p,
            r=r,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            d_digit=d_digit,
            dropout=dropout,
        )
        self.head = AnomalyHead(d_model=d_model, hidden_dim=head_hidden, dropout=dropout)

    def encode(self, digits: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(digits)
        h, _ = self.encoder(digits, x, src_key_padding_mask=padding_mask)
        return h

    def forward_with_features(self, digits: torch.Tensor, padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(digits, padding_mask=padding_mask)
        pooled = self.head.pool_hidden(hidden, padding_mask=padding_mask)
        logits = self.head.net(pooled).squeeze(-1)
        return logits, pooled

    def forward(self, digits: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self.forward_with_features(digits, padding_mask=padding_mask)
        return logits

    def forward_with_attention(self, digits: torch.Tensor, padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        x = self.embedding(digits)
        h, weights = self.encoder(digits, x, src_key_padding_mask=padding_mask)
        return self.head(h, padding_mask=padding_mask), weights

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def parameter_summary(self) -> str:
        total = self.count_parameters()
        return "\n".join(
            [
                f"PadicAttentionAnomalyDetector  p={self.p}  r={self.r}  d_model={self.d_model}",
                f"  HenselEmbedding : {sum(param.numel() for param in self.embedding.parameters()):>10,}",
                f"  AttentionEncoder: {sum(param.numel() for param in self.encoder.parameters()):>10,}",
                f"  AnomalyHead     : {sum(param.numel() for param in self.head.parameters()):>10,}",
                f"  -- Total        : {total:>10,}",
            ]
        )


def compare_hard_soft_valuation(digits_a: torch.Tensor, digits_b: torch.Tensor, valuation: SoftPadicValuation) -> dict[str, float]:
    if digits_a.shape != digits_b.shape:
        raise ValueError("digits_a and digits_b must have the same shape")
    hard = []
    for a, b in zip(digits_a, digits_b):
        hard.append(float((a == b).to(torch.int64).cumprod(dim=-1).sum().item()))
    soft = valuation(digits_a, digits_b).detach().cpu()
    hard_t = torch.tensor(hard, dtype=soft.dtype)
    if hard_t.numel() < 2:
        correlation = float("nan")
    else:
        hard_c = hard_t - hard_t.mean()
        soft_c = soft - soft.mean()
        denom = hard_c.norm() * soft_c.norm()
        correlation = float((hard_c @ soft_c / denom).item()) if float(denom) != 0.0 else float("nan")
    return {"correlation": correlation, "hard_mean": float(hard_t.mean().item()), "soft_mean": float(soft.mean().item())}

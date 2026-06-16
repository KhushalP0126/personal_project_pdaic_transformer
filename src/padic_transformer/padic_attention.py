"""Soft p-adic attention layers for anomaly detection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import AnomalyHead, HenselEmbedding
from .hensel import digits_to_int64


def _attention_sparsity(weights: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
    """Return the fraction of attention weights below `threshold`."""
    if weights.numel() == 0:
        return weights.new_tensor(0.0)
    return (weights < threshold).to(torch.float32).mean()


def _hard_prefix_matrix(digits: torch.Tensor) -> torch.Tensor:
    if digits.ndim != 3:
        raise ValueError("digits must have shape [batch, seq, r]")
    equal = digits.unsqueeze(2) == digits.unsqueeze(1)
    return equal.to(torch.int64).cumprod(dim=-1).sum(dim=-1)


def _safe_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have matching shapes")
    masked = values.masked_select(mask)
    if masked.numel() == 0:
        return values.new_tensor(0.0)
    return masked.mean()


def _finite_or_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(value), value, value.new_tensor(0.0))


def _prime_gaps(count: int) -> list[int]:
    if count <= 0:
        return []

    gaps: list[int] = []
    previous_prime = 2
    candidate = 3
    while len(gaps) < count:
        is_prime = True
        limit = int(math.isqrt(candidate))
        for factor in range(3, limit + 1, 2):
            if candidate % factor == 0:
                is_prime = False
                break
        if is_prime:
            gaps.append(candidate - previous_prime)
            previous_prime = candidate
        candidate += 2
    return gaps


def _subset_attention_metrics(weights: torch.Tensor, hard_prefix: torch.Tensor, pair_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    same_cluster = pair_mask & (hard_prefix > 0)
    diff_cluster = pair_mask & (hard_prefix == 0)
    same_cluster_attention = _safe_masked_mean(weights, same_cluster)
    diff_cluster_attention = _safe_masked_mean(weights, diff_cluster)
    hierarchy_gap = same_cluster_attention - diff_cluster_attention

    flat_weights = weights.masked_select(pair_mask)
    flat_prefix = hard_prefix.masked_select(pair_mask)
    if flat_weights.numel() < 2:
        corr = weights.new_tensor(0.0)
    else:
        centered_weights = flat_weights - flat_weights.mean()
        centered_prefix = flat_prefix - flat_prefix.mean()
        denom = centered_weights.norm() * centered_prefix.norm()
        if float(denom.item()) == 0.0:
            corr = weights.new_tensor(0.0)
        else:
            corr = (centered_weights @ centered_prefix) / denom

    return {
        "padic_attention_corr": corr,
        "same_cluster_attention": same_cluster_attention,
        "diff_cluster_attention": diff_cluster_attention,
        "hierarchy_gap": hierarchy_gap,
    }


def _attention_hierarchy_metrics(
    weights: torch.Tensor,
    digits: torch.Tensor,
    p: int,
    key_padding_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if weights.ndim != 3:
        raise ValueError("weights must have shape [batch, seq, seq]")
    hard_prefix = _hard_prefix_matrix(digits).to(dtype=weights.dtype)
    batch, seq, _ = hard_prefix.shape
    valid = torch.ones((batch, seq), dtype=torch.bool, device=weights.device)
    if key_padding_mask is not None:
        valid = ~key_padding_mask

    valid_pairs = valid.unsqueeze(2) & valid.unsqueeze(1)
    pair_mask = valid_pairs & ~torch.eye(seq, dtype=torch.bool, device=weights.device).unsqueeze(0)

    metrics = _subset_attention_metrics(weights, hard_prefix, pair_mask)
    for depth in (1, 2, 4):
        same_depth = pair_mask & (hard_prefix >= depth)
        diff_depth = pair_mask & (hard_prefix < depth)
        same_depth_attention = _safe_masked_mean(weights, same_depth)
        diff_depth_attention = _safe_masked_mean(weights, diff_depth)
        metrics[f"attn_gap_depth{depth}"] = same_depth_attention - diff_depth_attention
    try:
        flat_weights = weights.masked_select(pair_mask).float()
        flat_ids = digits_to_int64(digits.view(-1, digits.shape[-1]), p=p)
        flat_ids_mat = flat_ids.view(digits.shape[0], digits.shape[1])
        id_diff = (flat_ids_mat.unsqueeze(2) - flat_ids_mat.unsqueeze(1)).abs().float()
        flat_id_diff = id_diff.masked_select(pair_mask)
        if flat_weights.numel() < 2 or flat_id_diff.std() < 1e-6:
            tp_corr = weights.new_tensor(0.0)
            tp_gap = weights.new_tensor(0.0)
        else:
            cw = flat_weights - flat_weights.mean()
            cd = flat_id_diff - flat_id_diff.mean()
            denom = cw.norm() * cd.norm()
            tp_corr = -(cw @ cd) / denom if float(denom.item()) > 0 else weights.new_tensor(0.0)
            median_diff = flat_id_diff.median()
            close_mask_flat = flat_id_diff <= median_diff
            far_mask_flat = ~close_mask_flat
            close_attn = flat_weights[close_mask_flat].mean() if close_mask_flat.any() else weights.new_tensor(0.0)
            far_attn = flat_weights[far_mask_flat].mean() if far_mask_flat.any() else weights.new_tensor(0.0)
            tp_gap = close_attn - far_attn
        metrics["twin_prime_stress_padic_attention_corr"] = _finite_or_zero(tp_corr)
        metrics["twin_prime_stress_hierarchy_gap"] = _finite_or_zero(tp_gap)
        metrics["twin_prime_stress_same_cluster_attention"] = _finite_or_zero(
            weights.masked_select(pair_mask & (hard_prefix > 0)).mean()
            if (pair_mask & (hard_prefix > 0)).any()
            else weights.new_tensor(0.0)
        )
        metrics["twin_prime_stress_diff_cluster_attention"] = _finite_or_zero(
            weights.masked_select(pair_mask & (hard_prefix == 0)).mean()
            if (pair_mask & (hard_prefix == 0)).any()
            else weights.new_tensor(0.0)
        )
    except OverflowError:
        metrics["twin_prime_stress_padic_attention_corr"] = weights.new_tensor(0.0)
        metrics["twin_prime_stress_hierarchy_gap"] = weights.new_tensor(0.0)
        metrics["twin_prime_stress_same_cluster_attention"] = weights.new_tensor(0.0)
        metrics["twin_prime_stress_diff_cluster_attention"] = weights.new_tensor(0.0)
    metrics = {key: _finite_or_zero(value) for key, value in metrics.items()}
    metrics["attention_sparsity"] = _finite_or_zero(_attention_sparsity(weights.masked_select(valid_pairs)))
    return metrics


class SoftPadicValuation(nn.Module):
    """Differentiable approximation of shared low-order Hensel prefix length."""

    def __init__(
        self,
        p: int,
        r: int,
        d_digit: int = 16,
        temperature: float = 1.0,
        learn_temp: bool = True,
        eps: float = 1e-6,
        diversity_weight: float = 0.01,
        temperature_decay: float = 0.05,
        hard_match: bool = False,
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
        self.diversity_weight = diversity_weight
        self.temperature_decay = temperature_decay
        self.hard_match = hard_match

        gap_pattern = _prime_gaps(r)
        init_temps = torch.tensor(
            [max(1e-6, temperature * (1.0 + temperature_decay * float(gap))) for gap in gap_pattern],
            dtype=torch.float32,
        )
        if not hard_match:
            self.digit_embeddings = nn.ModuleList([nn.Embedding(p, d_digit) for _ in range(r)])
        else:
            self.digit_embeddings = nn.ModuleList()  # unused in hard mode
        if learn_temp:
            self.log_temperature = nn.Parameter(init_temps.log())
        else:
            self.register_buffer("log_temperature", init_temps.log())
        self.prefix_weights = nn.Parameter(torch.ones(r))
        if not hard_match:
            self._reset_parameters()

    def _reset_parameters(self) -> None:
        for emb in self.digit_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=self.d_digit ** -0.5)

    def temperature_stats(self) -> dict[str, float]:
        temps = self.log_temperature.exp().detach()
        temp_std = temps.std(unbiased=False)
        return {
            "temp_mean": float(temps.mean().item()),
            "temp_std": float(temp_std.item()),
            "temp_min": float(temps.min().item()),
            "temp_max": float(temps.max().item()),
            "temp_pos0": float(temps[0].item()),
            "temp_pos_last": float(temps[-1].item()),
            "collapsed": bool((temp_std < 0.1 * temps.mean()).item()),
        }

    def temperature_diversity_loss(self) -> torch.Tensor:
        temps = self.log_temperature.exp()
        mean = temps.mean()
        std = temps.std(unbiased=False)
        cv = std / (mean + self.eps)
        target_cv = 0.3
        return self.diversity_weight * F.relu(target_cv - cv)

    def _soft_match_at_position(self, digits_a: torch.Tensor, digits_b: torch.Tensor, position: int) -> torch.Tensor:
        if self.hard_match:
            # Exact digit equality: 1.0 if equal, 0.0 if not.
            # Use a steep sigmoid to keep gradients flowing through temperature.
            match = (digits_a == digits_b).to(torch.float32)
            tau = self.log_temperature[position].exp().clamp(min=0.01, max=20.0)
            # Map match ∈ {0, 1} to sigmoid input: match=1 → +tau, match=0 → −tau
            return torch.sigmoid(tau * (match - 0.5))
        emb = self.digit_embeddings[position]
        ea = emb(digits_a)
        eb = emb(digits_b)
        tau = self.log_temperature[position].exp().clamp(min=0.01, max=20.0)
        cos_sim = F.cosine_similarity(ea, eb, dim=-1)
        similarity = 0.5 * (1.0 + cos_sim)
        return torch.sigmoid(tau * (similarity - 0.5))

    def forward(self, digits_a: torch.Tensor, digits_b: torch.Tensor) -> torch.Tensor:
        if digits_a.shape != digits_b.shape:
            raise ValueError("digits_a and digits_b must have the same shape")
        if digits_a.shape[-1] != self.r:
            raise ValueError(f"Last dimension must be r={self.r}")

        prefixes = []
        running = None
        for pos in range(self.r):
            match = self._soft_match_at_position(digits_a[..., pos], digits_b[..., pos], pos)
            running = match if running is None else running * match
            prefixes.append(running)
        prefix = torch.stack(prefixes, dim=-1)
        weights = F.softplus(self.prefix_weights)
        soft_val = (prefix * weights).sum(dim=-1)
        return soft_val / (weights.sum() + self.eps)

    def pairwise(self, digits: torch.Tensor) -> torch.Tensor:
        """Return soft shared-prefix scores for all pairs in each window.

        This computes one digit position at a time and avoids materializing a
        full [batch, seq, seq, r] expanded integer tensor inside each attention
        head.
        """
        if digits.ndim != 3 or digits.shape[-1] != self.r:
            raise ValueError(f"digits must have shape [batch, seq, r={self.r}]")

        batch, seq, _ = digits.shape
        weights = F.softplus(self.prefix_weights).to(torch.float32)
        running_log = digits.new_zeros((batch, seq, seq), dtype=torch.float32)
        soft_val = digits.new_zeros((batch, seq, seq), dtype=torch.float32)
        for pos in range(self.r):
            tau = self.log_temperature[pos].exp().clamp(min=0.01, max=20.0).to(torch.float32)
            if self.hard_match:
                d = digits[..., pos]
                match = (d.unsqueeze(2) == d.unsqueeze(1)).to(torch.float32)
                similarity = match
            else:
                emb = self.digit_embeddings[pos](digits[..., pos]).to(torch.float32)
                cos_sim = F.cosine_similarity(emb.unsqueeze(2), emb.unsqueeze(1), dim=-1)
                similarity = 0.5 * (1.0 + cos_sim)
            running_log = running_log + F.logsigmoid(tau * (similarity - 0.5))
            soft_val = soft_val + torch.exp(running_log) * weights[pos]
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
        hard_match: bool = False,
    ) -> None:
        super().__init__()
        self.valuation = SoftPadicValuation(p=p, r=r, d_digit=d_digit, hard_match=hard_match)
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.query_proj = nn.Linear(d_model, d_head)
        self.key_proj = nn.Linear(d_model, d_head)
        self.padic_gate = nn.Parameter(torch.tensor(0.0))
        self.gate_regularization_weight = 0.001
        self.value_proj = nn.Linear(d_model, d_head)
        self.dropout = nn.Dropout(dropout)

    def gate_regularization_loss(self) -> torch.Tensor:
        gate = self.padic_gate.sigmoid()
        return self.gate_regularization_weight * (gate - 0.5).square()

    def forward(
        self,
        digits: torch.Tensor,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_metrics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if digits.ndim != 3 or x.ndim != 3:
            raise ValueError("digits and x must be [batch, seq, ...]")
        if digits.shape[:2] != x.shape[:2]:
            raise ValueError("digits and x must have matching batch/seq dimensions")
        if key_padding_mask is not None:
            if key_padding_mask.shape != digits.shape[:2]:
                raise ValueError(
                    f"key_padding_mask must have shape {tuple(digits.shape[:2])}, "
                    f"got {tuple(key_padding_mask.shape)}"
                )
            if bool(key_padding_mask.all(dim=1).any().item()):
                raise ValueError("fully padded samples are not supported")

        raw = self.valuation.pairwise(digits).to(dtype=x.dtype)
        scale = self.logit_scale.clamp(0.1, 20.0)
        q = self.query_proj(x)
        k = self.key_proj(x)
        content_logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        content_std = content_logits.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-3)
        content_logits = content_logits / content_std
        padic_logits = scale * raw
        padic_logits = padic_logits - padic_logits.mean(dim=-1, keepdim=True)
        padic_std = padic_logits.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-3)
        padic_logits = padic_logits / padic_std
        logits = content_logits + self.padic_gate.sigmoid() * padic_logits

        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask.unsqueeze(1), torch.finfo(logits.dtype).min)

        weights = torch.softmax(logits, dim=-1)
        if key_padding_mask is not None:
            key_valid = (~key_padding_mask).unsqueeze(1).to(weights.dtype)
            weights = weights * key_valid
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        values = self.value_proj(x)
        out = torch.bmm(weights, values)
        out = self.dropout(out)
        if key_padding_mask is not None:
            out = out * (~key_padding_mask).unsqueeze(-1).to(out.dtype)
        if not return_metrics:
            return out, weights
        metrics = {"padic_gate": self.padic_gate.sigmoid().detach()}
        metrics.update(_attention_hierarchy_metrics(weights, digits, self.valuation.p, key_padding_mask))
        return out, weights, metrics


class PadicMultiHeadAttention(nn.Module):
    def __init__(
        self,
        p: int,
        r: int,
        d_model: int,
        n_heads: int,
        d_digit: int = 16,
        dropout: float = 0.1,
        hard_match: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.heads = nn.ModuleList(
            [
                PadicAttentionHead(p=p, r=r, d_model=d_model, d_head=d_model // n_heads, d_digit=d_digit, dropout=dropout, hard_match=hard_match)
                for _ in range(n_heads)
            ]
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        digits: torch.Tensor,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_metrics: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        outputs = []
        weights = []
        metric_lists: dict[str, list[torch.Tensor]] = {
            "attention_sparsity": [],
            "padic_attention_corr": [],
            "same_cluster_attention": [],
            "diff_cluster_attention": [],
            "hierarchy_gap": [],
            "twin_prime_stress_padic_attention_corr": [],
            "twin_prime_stress_same_cluster_attention": [],
            "twin_prime_stress_diff_cluster_attention": [],
            "twin_prime_stress_hierarchy_gap": [],
            "attn_gap_depth1": [],
            "attn_gap_depth2": [],
            "attn_gap_depth4": [],
            "padic_gate": [],
        }
        for head in self.heads:
            if return_metrics:
                out, attn, metrics = head(
                    digits,
                    x,
                    key_padding_mask=key_padding_mask,
                    return_metrics=True,
                )
                for key in metric_lists:
                    metric_lists[key].append(metrics[key])
            else:
                out, attn = head(digits, x, key_padding_mask=key_padding_mask)
            outputs.append(out)
            weights.append(attn)
        out = self.out_proj(torch.cat(outputs, dim=-1))
        if key_padding_mask is not None:
            out = out * (~key_padding_mask).unsqueeze(-1).to(out.dtype)
        if not return_metrics:
            return out, weights
        metrics = {
            key: torch.stack(values).mean() if values else out.new_tensor(float("nan"))
            for key, values in metric_lists.items()
        }
        return out, weights, metrics


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
        hard_match: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = PadicMultiHeadAttention(p=p, r=r, d_model=d_model, n_heads=n_heads, d_digit=d_digit, dropout=dropout, hard_match=hard_match)
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
        return_metrics: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        if return_metrics:
            attn_out, weights, metrics = self.self_attn(
                digits,
                self.norm1(x),
                key_padding_mask=src_key_padding_mask,
                return_metrics=True,
            )
        else:
            attn_out, weights = self.self_attn(
                digits,
                self.norm1(x),
                key_padding_mask=src_key_padding_mask,
            )
        valid = None
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).unsqueeze(-1).to(x.dtype)
        x = x + self.dropout(attn_out)
        if valid is not None:
            x = x * valid
        x = self.norm2(x)
        x = x + self.dropout(self.ffn(x))
        if valid is not None:
            x = x * valid
        if not return_metrics:
            return x, weights
        return x, weights, metrics


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
        hard_match: bool = False,
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
                    hard_match=hard_match,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        digits: torch.Tensor,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        return_metrics: bool = False,
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]] | tuple[torch.Tensor, list[list[torch.Tensor]], dict[str, torch.Tensor]]:
        all_weights = []
        layer_metrics: dict[str, list[torch.Tensor]] = {
            "attention_sparsity": [],
            "padic_attention_corr": [],
            "same_cluster_attention": [],
            "diff_cluster_attention": [],
            "hierarchy_gap": [],
            "twin_prime_stress_padic_attention_corr": [],
            "twin_prime_stress_same_cluster_attention": [],
            "twin_prime_stress_diff_cluster_attention": [],
            "twin_prime_stress_hierarchy_gap": [],
            "attn_gap_depth1": [],
            "attn_gap_depth2": [],
            "attn_gap_depth4": [],
            "padic_gate": [],
        }
        for layer in self.layers:
            if return_metrics:
                x, weights, metrics = layer(
                    digits,
                    x,
                    src_key_padding_mask=src_key_padding_mask,
                    return_metrics=True,
                )
                for key in layer_metrics:
                    layer_metrics[key].append(metrics[key])
            else:
                x, weights = layer(digits, x, src_key_padding_mask=src_key_padding_mask)
            all_weights.append(weights)
        if not return_metrics:
            return x, all_weights
        metrics = {
            key: torch.stack(values).mean() if values else x.new_tensor(float("nan"))
            for key, values in layer_metrics.items()
        }
        return x, all_weights, metrics


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
        max_seq_len: int = 256,
        hard_match: bool = False,
        ) -> None:
        super().__init__()
        self.p = p
        self.r = r
        self.d_model = d_model
        self.embedding = HenselEmbedding(p=p, r=r, d_model=d_model, max_seq_len=max_seq_len, dropout=dropout)
        self.encoder = PadicAttentionEncoder(
            p=p,
            r=r,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            d_digit=d_digit,
            dropout=dropout,
            hard_match=hard_match,
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

    def forward_with_attention(
        self,
        digits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_metrics: bool = False,
        return_features: bool = False,
    ) -> (
        tuple[torch.Tensor, list[list[torch.Tensor]]]
        | tuple[torch.Tensor, list[list[torch.Tensor]], dict[str, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]
        | tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]], dict[str, torch.Tensor]]
    ):
        x = self.embedding(digits)
        if return_metrics:
            h, weights, metrics = self.encoder(
                digits,
                x,
                src_key_padding_mask=padding_mask,
                return_metrics=True,
            )
            pooled = self.head.pool_hidden(h, padding_mask=padding_mask)
            logits = self.head.net(pooled).squeeze(-1)
            if return_features:
                return logits, pooled, weights, metrics
            return logits, weights, metrics
        h, weights = self.encoder(digits, x, src_key_padding_mask=padding_mask)
        pooled = self.head.pool_hidden(h, padding_mask=padding_mask)
        logits = self.head.net(pooled).squeeze(-1)
        if return_features:
            return logits, pooled, weights
        return logits, weights

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

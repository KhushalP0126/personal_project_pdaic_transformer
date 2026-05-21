"""Shared architectural fixes for the p-adic transformer models."""

from __future__ import annotations

import warnings
from collections import deque

import torch
import torch.nn as nn


class HenselEmbeddingWithPosition(nn.Module):
    """Digit-wise Hensel embedding with optional learnable positional encoding."""

    def __init__(
        self,
        p: int,
        r: int,
        d_model: int,
        max_seq_len: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if p < 2:
            raise ValueError("p must be >= 2")
        if r < 1:
            raise ValueError("r must be >= 1")
        if d_model < 1:
            raise ValueError("d_model must be >= 1")
        if max_seq_len < 0:
            raise ValueError("max_seq_len must be >= 0")

        self.p = p
        self.r = r
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.digit_embeds = nn.ModuleList([nn.Embedding(p, d_model) for _ in range(r)])
        self.pos_embed = nn.Embedding(max_seq_len, d_model) if max_seq_len > 0 else None
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for embed in self.digit_embeds:
            nn.init.normal_(embed.weight, mean=0.0, std=self.d_model ** -0.5)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.01)

    def forward(self, digits: torch.Tensor) -> torch.Tensor:
        if digits.ndim != 3 or digits.shape[-1] != self.r:
            raise ValueError(
                f"digits must be shape [batch, seq, {self.r}], got {tuple(digits.shape)}"
            )
        seq_len = digits.shape[1]
        if self.pos_embed is not None and seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}"
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

        if self.pos_embed is not None:
            positions = torch.arange(seq_len, device=digits.device)
            out = out + self.pos_embed(positions).unsqueeze(0)
        return self.dropout(out)


def _temperature_modules(model: nn.Module) -> list[nn.Module]:
    modules: list[nn.Module] = []
    for module in model.modules():
        if hasattr(module, "temperature_stats") and hasattr(module, "temperature_diversity_loss"):
            modules.append(module)
    return modules


def compute_diversity_regularization(model: nn.Module) -> torch.Tensor:
    """Sum all temperature diversity penalties in a model."""
    modules = _temperature_modules(model)
    if not modules:
        first_param = next(model.parameters(), None)
        if first_param is None:
            return torch.tensor(0.0)
        return first_param.new_tensor(0.0)

    loss = None
    for module in modules:
        term = module.temperature_diversity_loss()
        loss = term if loss is None else loss + term
    assert loss is not None
    return loss


def log_temperature_health(model: nn.Module, epoch: int) -> dict[str, float]:
    """Print a compact temperature health summary for all valuation modules."""
    modules = _temperature_modules(model)
    if not modules:
        return {}

    stats = [module.temperature_stats() for module in modules]
    mean_temp = sum(item["temp_mean"] for item in stats) / len(stats)
    mean_std = sum(item["temp_std"] for item in stats) / len(stats)
    collapsed = any(item["collapsed"] for item in stats)
    print(
        f"  Temperature health epoch {epoch}: mean={mean_temp:.4f} std={mean_std:.4f}"
        + (" COLLAPSED" if collapsed else "")
    )
    if collapsed:
        warnings.warn(
            "temperature collapse detected in p-adic valuation modules",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "temp_mean": mean_temp,
        "temp_std": mean_std,
        "collapsed": float(collapsed),
    }


def quantize_dynamic_model(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to linear layers for CPU inference."""
    if next(model.parameters(), None) is not None and next(model.parameters()).device.type != "cpu":
        raise ValueError("quantize_dynamic_model expects a CPU model")
    if hasattr(torch.backends, "quantized"):
        engine = getattr(torch.backends.quantized, "engine", None)
        if engine in (None, "none", "NoQEngine"):
            supported = getattr(torch.backends.quantized, "supported_engines", [])
            if supported:
                torch.backends.quantized.engine = supported[0]
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


class StreamingWindowScorer:
    """Score a token stream incrementally with a fixed-size window detector."""

    def __init__(self, model: nn.Module, window_size: int) -> None:
        self.model = model
        self.window_size = window_size
        self._buffer: deque[torch.Tensor] = deque(maxlen=window_size)

    def push(self, token_digits: torch.Tensor) -> torch.Tensor | None:
        if token_digits.ndim != 2:
            raise ValueError("token_digits must be [seq, r]")
        logits = None
        for token in token_digits:
            self._buffer.append(token.detach())
            if len(self._buffer) == self.window_size:
                window = torch.stack(list(self._buffer), dim=0).unsqueeze(0)
                with torch.no_grad():
                    logits = self.model(window).squeeze(0)
        return logits

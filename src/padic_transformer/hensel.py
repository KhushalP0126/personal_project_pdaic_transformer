"""PyTorch Hensel-code utilities.

Digits are stored least-significant first. For p-adic distance this means closeness is
the length of the matching prefix starting at digit index 0.
"""

from __future__ import annotations

import torch


INT64_MAX = torch.iinfo(torch.int64).max


def _require_digit_tensor(digits: torch.Tensor, p: int) -> torch.Tensor:
    arr = torch.as_tensor(digits)
    if arr.ndim == 0:
        raise ValueError("digits must have at least one dimension")
    if p < 2:
        raise ValueError("p must be >= 2")
    if bool(torch.any(arr < 0).item()) or bool(torch.any(arr >= p).item()):
        raise ValueError("digits must be in the range [0, p)")
    return arr.to(dtype=torch.int64)


def int64_to_digits(values: torch.Tensor, p: int, r: int) -> torch.Tensor:
    """Convert non-negative int64 residues to fixed-width Hensel digits."""
    if p < 2:
        raise ValueError("p must be >= 2")
    if not 1 <= r <= 64:
        raise ValueError("r must be in [1, 64]")

    work = torch.as_tensor(values, dtype=torch.int64).clone()
    if bool(torch.any(work < 0).item()):
        raise ValueError("values must be non-negative")
    out = torch.empty((*work.shape, r), dtype=torch.int64, device=work.device)
    for idx in range(r):
        out[..., idx] = work.remainder(p)
        work = torch.div(work, p, rounding_mode="floor")
    return out


def digits_to_int64(digits: torch.Tensor, p: int) -> torch.Tensor:
    """Pack fixed-width Hensel digits into int64 residues when the width fits."""
    arr = _require_digit_tensor(digits, p)
    r = arr.shape[-1]
    if pow(p, r) - 1 > int(INT64_MAX):
        raise OverflowError(f"p**r does not fit int64 for p={p}, r={r}")

    out = torch.zeros(arr.shape[:-1], dtype=torch.int64, device=arr.device)
    place = 1
    for idx in range(r):
        out += arr[..., idx] * place
        place *= p
    return out


def carry_left_add(left: torch.Tensor, right: torch.Tensor, p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Add two Hensel digit arrays with carry propagation toward higher indices.

    Returns `(digits, overflow)` where `overflow` is the carry after the highest
    retained digit. Broadcasting follows PyTorch rules except for the final digit axis.
    """
    a = _require_digit_tensor(left, p)
    b = _require_digit_tensor(right, p).to(device=a.device, dtype=torch.int64)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError("left and right must have the same precision axis")

    a, b = torch.broadcast_tensors(a, b)
    out = torch.empty_like(a)
    carry = torch.zeros(a.shape[:-1], dtype=torch.int64, device=a.device)
    for idx in range(a.shape[-1]):
        total = a[..., idx] + b[..., idx] + carry
        out[..., idx] = total.remainder(p)
        carry = torch.div(total, p, rounding_mode="floor")
    return out, carry


def shared_prefix_valuation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Count shared low-order Hensel digits for each pair of rows."""
    a = torch.as_tensor(left)
    b = torch.as_tensor(right, device=a.device)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)

    equal = a == b
    prefix = equal.to(torch.int64).cumprod(dim=-1)
    return prefix.sum(dim=-1).to(torch.int64)

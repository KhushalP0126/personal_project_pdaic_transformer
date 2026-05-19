"""Smoke tests for the soft p-adic attention module."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from padic_transformer.padic_attention import (
    PadicAttentionAnomalyDetector,
    PadicAttentionHead,
    PadicMultiHeadAttention,
    PadicTransformerLayer,
    SoftPadicValuation,
)


class TestSoftPadicValuation(unittest.TestCase):
    def test_shapes(self) -> None:
        valuation = SoftPadicValuation(p=3, r=8, d_digit=8)
        a = torch.randint(0, 3, (4, 8))
        b = torch.randint(0, 3, (4, 8))
        out = valuation(a, b)
        self.assertEqual(out.shape, (4,))


class TestPadicAttentionHead(unittest.TestCase):
    def test_output_shape(self) -> None:
        head = PadicAttentionHead(p=3, r=8, d_model=32, d_head=16, d_digit=8)
        digits = torch.randint(0, 3, (2, 5, 8))
        x = torch.randn(2, 5, 32)
        out, weights = head(digits, x)
        self.assertEqual(out.shape, (2, 5, 16))
        self.assertEqual(weights.shape, (2, 5, 5))


class TestPadicMultiHeadAttention(unittest.TestCase):
    def test_output_shape(self) -> None:
        mha = PadicMultiHeadAttention(p=3, r=8, d_model=32, n_heads=4, d_digit=8)
        digits = torch.randint(0, 3, (2, 5, 8))
        x = torch.randn(2, 5, 32)
        out, weights = mha(digits, x)
        self.assertEqual(out.shape, (2, 5, 32))
        self.assertEqual(len(weights), 4)


class TestPadicAttentionAnomalyDetector(unittest.TestCase):
    def test_forward_with_attention(self) -> None:
        model = PadicAttentionAnomalyDetector(
            p=3,
            r=8,
            d_model=32,
            n_heads=4,
            n_layers=2,
            ffn_dim=64,
            head_hidden=16,
            d_digit=8,
        )
        digits = torch.randint(0, 3, (2, 5, 8))
        logits, attn = model.forward_with_attention(digits)
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(len(attn), 2)
        self.assertEqual(len(attn[0]), 4)

    def test_forward_with_features(self) -> None:
        model = PadicAttentionAnomalyDetector(
            p=3,
            r=8,
            d_model=32,
            n_heads=4,
            n_layers=2,
            ffn_dim=64,
            head_hidden=16,
            d_digit=8,
        )
        digits = torch.randint(0, 3, (2, 5, 8))
        logits, features = model.forward_with_features(digits)
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(features.shape, (2, 32))


if __name__ == "__main__":
    unittest.main()

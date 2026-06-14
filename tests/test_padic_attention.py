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

    def test_prime_gap_temperature_initialization_is_bumpy(self) -> None:
        valuation = SoftPadicValuation(p=3, r=5, d_digit=8, temperature=1.0, temperature_decay=0.1)
        temps = valuation.log_temperature.exp()
        expected = torch.tensor([1.1, 1.2, 1.2, 1.4, 1.2], dtype=temps.dtype)
        torch.testing.assert_close(temps, expected, atol=1e-6, rtol=0.0)

    def test_r1_temperature_stats_and_loss_are_finite(self) -> None:
        valuation = SoftPadicValuation(p=3, r=1, d_digit=8)
        stats = valuation.temperature_stats()
        self.assertEqual(stats["temp_std"], 0.0)
        for value in stats.values():
            if isinstance(value, bool):
                continue
            self.assertTrue(torch.isfinite(torch.tensor(value)).item())
        loss = valuation.temperature_diversity_loss()
        self.assertTrue(torch.isfinite(loss).item())

    def test_seq_len_one_logits_are_finite(self) -> None:
        head = PadicAttentionHead(p=3, r=8, d_model=32, d_head=16, d_digit=8)
        digits = torch.randint(0, 3, (2, 1, 8))
        x = torch.randn(2, 1, 32)
        out, weights, metrics = head(digits, x, return_metrics=True)
        self.assertEqual(out.shape, (2, 1, 16))
        self.assertEqual(weights.shape, (2, 1, 1))
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertTrue(torch.isfinite(weights).all().item())
        for value in metrics.values():
            self.assertTrue(torch.isfinite(value).item())


class TestPadicAttentionHead(unittest.TestCase):
    def test_output_shape(self) -> None:
        head = PadicAttentionHead(p=3, r=8, d_model=32, d_head=16, d_digit=8)
        digits = torch.randint(0, 3, (2, 5, 8))
        x = torch.randn(2, 5, 32)
        out, weights = head(digits, x)
        self.assertEqual(out.shape, (2, 5, 16))
        self.assertEqual(weights.shape, (2, 5, 5))

    def test_padded_queries_zero_output(self) -> None:
        head = PadicAttentionHead(p=3, r=8, d_model=32, d_head=16, d_digit=8)
        digits = torch.randint(0, 3, (2, 5, 8))
        x = torch.randn(2, 5, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[:, -1] = True
        out, _ = head(digits, x, key_padding_mask=mask)
        torch.testing.assert_close(out[:, -1], torch.zeros_like(out[:, -1]), atol=1e-6, rtol=0.0)

    def test_metrics_include_hierarchy_signals(self) -> None:
        head = PadicAttentionHead(p=3, r=8, d_model=32, d_head=16, d_digit=8)
        digits = torch.randint(0, 3, (2, 5, 8))
        x = torch.randn(2, 5, 32)
        _, _, metrics = head(digits, x, return_metrics=True)
        for key in (
            "attention_sparsity",
            "padic_attention_corr",
            "same_cluster_attention",
            "diff_cluster_attention",
            "hierarchy_gap",
            "twin_prime_stress_padic_attention_corr",
            "twin_prime_stress_same_cluster_attention",
            "twin_prime_stress_diff_cluster_attention",
            "twin_prime_stress_hierarchy_gap",
            "attn_gap_depth1",
            "attn_gap_depth2",
            "attn_gap_depth4",
            "padic_gate",
        ):
            self.assertIn(key, metrics)
        gate = float(metrics["padic_gate"].item())
        self.assertAlmostEqual(gate, 0.5, places=6)

    def test_attention_seq_len_one_is_finite(self) -> None:
        head = PadicAttentionHead(p=3, r=8, d_model=32, d_head=16, d_digit=8)
        digits = torch.randint(0, 3, (2, 1, 8))
        x = torch.randn(2, 1, 32)
        out, weights = head(digits, x)
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertTrue(torch.isfinite(weights).all().item())

    def test_attention_rejects_fully_padded_sample(self) -> None:
        head = PadicAttentionHead(p=3, r=8, d_model=32, d_head=16, d_digit=8)
        digits = torch.randint(0, 3, (2, 5, 8))
        x = torch.randn(2, 5, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[0, :] = True

        with self.assertRaises(ValueError):
            head(digits, x, key_padding_mask=mask)

    def test_attention_weights_renormalize_after_key_mask(self) -> None:
        head = PadicAttentionHead(p=3, r=8, d_model=32, d_head=16, d_digit=8)
        digits = torch.randint(0, 3, (2, 5, 8))
        x = torch.randn(2, 5, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[:, -2:] = True

        _, weights = head(digits, x, key_padding_mask=mask)

        valid_queries = ~mask
        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(
            row_sums[valid_queries],
            torch.ones_like(row_sums[valid_queries]),
            atol=1e-6,
            rtol=0.0,
        )


class TestPadicMultiHeadAttention(unittest.TestCase):
    def test_output_shape(self) -> None:
        mha = PadicMultiHeadAttention(p=3, r=8, d_model=32, n_heads=4, d_digit=8)
        digits = torch.randint(0, 3, (2, 5, 8))
        x = torch.randn(2, 5, 32)
        out, weights = mha(digits, x)
        self.assertEqual(out.shape, (2, 5, 32))
        self.assertEqual(len(weights), 4)

    def test_padded_queries_zero_after_out_projection(self) -> None:
        mha = PadicMultiHeadAttention(p=3, r=8, d_model=32, n_heads=4, d_digit=8)
        digits = torch.randint(0, 3, (2, 5, 8))
        x = torch.randn(2, 5, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[:, -2:] = True
        out, _ = mha(digits, x, key_padding_mask=mask)
        torch.testing.assert_close(out[:, -2:], torch.zeros_like(out[:, -2:]), atol=1e-6, rtol=0.0)


class TestAnomalyHead(unittest.TestCase):
    def test_all_padded_samples_are_rejected(self) -> None:
        from padic_transformer.model import AnomalyHead

        head = AnomalyHead(d_model=16, hidden_dim=8)
        hidden = torch.randn(2, 4, 16)
        mask = torch.ones(2, 4, dtype=torch.bool)
        with self.assertRaises(ValueError):
            head.pool_hidden(hidden, padding_mask=mask)


class TestPadicTransformerLayer(unittest.TestCase):
    def test_padded_tokens_stay_zero_after_attention_and_ffn(self) -> None:
        layer = PadicTransformerLayer(p=3, r=8, d_model=32, n_heads=4, ffn_dim=64, d_digit=8)
        digits = torch.randint(0, 3, (2, 5, 8))
        x = torch.randn(2, 5, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[:, -2:] = True
        out, _ = layer(digits, x, src_key_padding_mask=mask)
        torch.testing.assert_close(out[:, -2:], torch.zeros_like(out[:, -2:]), atol=1e-6, rtol=0.0)


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

    def test_forward_with_attention_metrics(self) -> None:
        model = PadicAttentionAnomalyDetector(
            p=5,
            r=8,
            d_model=32,
            n_heads=4,
            n_layers=2,
            ffn_dim=64,
            head_hidden=16,
            d_digit=8,
        )
        digits = torch.randint(0, 5, (2, 5, 8))
        logits, attn, metrics = model.forward_with_attention(digits, return_metrics=True)
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(len(attn), 2)
        for key in (
            "attention_sparsity",
            "padic_attention_corr",
            "same_cluster_attention",
            "diff_cluster_attention",
            "hierarchy_gap",
            "twin_prime_stress_padic_attention_corr",
            "twin_prime_stress_same_cluster_attention",
            "twin_prime_stress_diff_cluster_attention",
            "twin_prime_stress_hierarchy_gap",
            "attn_gap_depth1",
            "attn_gap_depth2",
            "attn_gap_depth4",
            "padic_gate",
        ):
            self.assertIn(key, metrics)
        sparsity = float(metrics["attention_sparsity"].item())
        self.assertGreaterEqual(sparsity, 0.0)
        self.assertLessEqual(sparsity, 1.0)
        self.assertTrue(torch.isfinite(metrics["twin_prime_stress_hierarchy_gap"]).item())

    def test_forward_with_attention_features(self) -> None:
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
        logits, features, attn, metrics = model.forward_with_attention(
            digits,
            return_metrics=True,
            return_features=True,
        )
        self.assertEqual(logits.shape, (2,))
        self.assertEqual(features.shape, (2, 64))
        self.assertEqual(len(attn), 2)
        self.assertIn("hierarchy_gap", metrics)

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
        self.assertEqual(features.shape, (2, 64))


if __name__ == "__main__":
    unittest.main()

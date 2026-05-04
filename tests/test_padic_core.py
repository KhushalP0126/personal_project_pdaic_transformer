from __future__ import annotations

import unittest

import torch

from padic_transformer.config import BenchmarkConfig, is_prime
from padic_transformer.hensel import carry_left_add, digits_to_int64, int64_to_digits
from padic_transformer.ultrametric import (
    generate_clustered_hensel_dataset,
    nearest_center_accuracy,
    ultrametric_violation_rate,
)


class PadicCoreTests(unittest.TestCase):
    def test_prime_validation(self) -> None:
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(5))
        self.assertFalse(is_prime(9))

    def test_hensel_round_trip(self) -> None:
        values = torch.tensor([0, 1, 7, 42, 255], dtype=torch.int64)
        digits = int64_to_digits(values, p=3, r=8)
        restored = digits_to_int64(digits, p=3)
        torch.testing.assert_close(restored, values, rtol=0, atol=0)

    def test_carry_left_add(self) -> None:
        left = torch.tensor([[2, 2, 0]], dtype=torch.int64)
        right = torch.tensor([[1, 0, 2]], dtype=torch.int64)
        digits, overflow = carry_left_add(left, right, p=3)
        torch.testing.assert_close(digits, torch.tensor([[0, 0, 0]], dtype=torch.int64))
        torch.testing.assert_close(overflow, torch.tensor([1], dtype=torch.int64))

    def test_generated_dataset_is_ultrametric(self) -> None:
        config = BenchmarkConfig(
            p=3,
            r=8,
            samples=512,
            classes=8,
            tokens_per_class=64,
            triplets=2000,
        )
        dataset = generate_clustered_hensel_dataset(config)
        rate, count = ultrametric_violation_rate(
            dataset.token_digits,
            triplets=config.triplets,
            seed=config.seed,
        )
        self.assertEqual(count, 0)
        self.assertEqual(rate, 0.0)

    def test_nearest_center_accuracy_has_signal(self) -> None:
        config = BenchmarkConfig(
            p=5,
            r=8,
            samples=512,
            classes=8,
            tokens_per_class=64,
        )
        dataset = generate_clustered_hensel_dataset(config)
        accuracy = nearest_center_accuracy(
            dataset.token_digits,
            dataset.token_labels,
            dataset.center_digits,
        )
        self.assertGreaterEqual(accuracy, 0.95)


if __name__ == "__main__":
    unittest.main()

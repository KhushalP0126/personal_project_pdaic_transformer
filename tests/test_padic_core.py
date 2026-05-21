from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from padic_transformer.config import BenchmarkConfig, is_prime
from padic_transformer.hensel import (
    carry_left_add,
    digits_to_int64,
    formal_power_series_coefficients,
    int64_to_digits,
)
from padic_transformer.ultrametric import (
    generate_clustered_hensel_dataset,
    map_raw_indices_excluding_pair,
    nearest_center_accuracy,
    sample_distinct_triplet_indices,
    ultrametric_violation_rate,
)


class PadicCoreTests(unittest.TestCase):
    def test_prime_validation(self) -> None:
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(5))
        self.assertFalse(is_prime(9))

    def test_config_validates_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkConfig(p=4, r=8)
        with self.assertRaises(ValueError):
            BenchmarkConfig(p=3, r=0)

    def test_hensel_round_trip(self) -> None:
        values = torch.tensor([0, 1, 7, 42, 255], dtype=torch.int64)
        digits = int64_to_digits(values, p=3, r=8)
        restored = digits_to_int64(digits, p=3)
        torch.testing.assert_close(restored, values, rtol=0, atol=0)

    def test_int64_to_digits_validate_rejects_truncation(self) -> None:
        with self.assertRaises(ValueError):
            int64_to_digits(torch.tensor([9], dtype=torch.int64), p=3, r=2, validate=True)

    def test_carry_left_add(self) -> None:
        left = torch.tensor([[2, 2, 0]], dtype=torch.int64)
        right = torch.tensor([[1, 0, 2]], dtype=torch.int64)
        digits, overflow = carry_left_add(left, right, p=3)
        torch.testing.assert_close(digits, torch.tensor([[0, 0, 0]], dtype=torch.int64))
        torch.testing.assert_close(overflow, torch.tensor([1], dtype=torch.int64))

    def test_formal_power_series_coefficients_supports_binary_and_odd_primes(self) -> None:
        left_binary = torch.tensor([[1, 1]], dtype=torch.int64)
        right_binary = torch.tensor([[1, 0]], dtype=torch.int64)
        binary = formal_power_series_coefficients(left_binary, right_binary, p=2)
        torch.testing.assert_close(binary, torch.tensor([[1, 1]], dtype=torch.int64))

        left_odd = torch.tensor([[1, 2]], dtype=torch.int64)
        right_odd = torch.tensor([[2, 1]], dtype=torch.int64)
        odd = formal_power_series_coefficients(left_odd, right_odd, p=3)
        torch.testing.assert_close(odd, torch.tensor([[2, 2]], dtype=torch.int64))

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

    def test_triplet_sampler_excludes_degenerate_rows(self) -> None:
        triplets = sample_distinct_triplet_indices(5, 1000, seed=123, device="cpu")
        self.assertTrue(torch.all(triplets[:, 0] != triplets[:, 1]).item())
        self.assertTrue(torch.all(triplets[:, 0] != triplets[:, 2]).item())
        self.assertTrue(torch.all(triplets[:, 1] != triplets[:, 2]).item())

    def test_excluding_pair_mapping_is_exact_bijection(self) -> None:
        for population in range(3, 9):
            raw = torch.arange(population - 2, dtype=torch.int64)
            for first in range(population):
                for second in range(population):
                    if first == second:
                        continue
                    lower, upper = sorted((first, second))
                    mapped = map_raw_indices_excluding_pair(
                        raw,
                        torch.full_like(raw, lower),
                        torch.full_like(raw, upper),
                    )
                    expected = torch.tensor(
                        [idx for idx in range(population) if idx not in {first, second}],
                        dtype=torch.int64,
                    )
                    torch.testing.assert_close(mapped, expected, rtol=0, atol=0)

    def test_excluding_pair_mapping_supports_mixed_batch(self) -> None:
        raw = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        lower = torch.tensor([1, 0, 2, 1], dtype=torch.int64)
        upper = torch.tensor([3, 2, 5, 4], dtype=torch.int64)
        mapped = map_raw_indices_excluding_pair(raw, lower, upper)
        expected = torch.tensor([0, 3, 3, 5], dtype=torch.int64)
        torch.testing.assert_close(mapped, expected, rtol=0, atol=0)

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

    def test_nearest_center_accuracy_accepts_permuted_centers(self) -> None:
        config = BenchmarkConfig(
            p=5,
            r=8,
            samples=512,
            classes=8,
            tokens_per_class=64,
        )
        dataset = generate_clustered_hensel_dataset(config)
        order = torch.tensor([2, 0, 1, 3, 4, 5, 6, 7], dtype=torch.int64)
        accuracy = nearest_center_accuracy(
            dataset.token_digits,
            dataset.token_labels,
            dataset.center_digits[order],
            dataset.center_labels[order],
        )
        self.assertGreaterEqual(accuracy, 0.95)


if __name__ == "__main__":
    unittest.main()

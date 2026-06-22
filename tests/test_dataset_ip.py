"""Tests for synthetic IPv4 prefix anomaly datasets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from padic_transformer.dataset_ip import (
    IPPrefixAnomalyDataset,
    IPPrefixDatasetConfig,
    binary_digits_to_int,
    int_to_ipv4,
    ipv4_to_binary_digits,
    ipv4_to_int,
)


class TestIPv4Digits(unittest.TestCase):
    def test_ipv4_roundtrip(self) -> None:
        value = ipv4_to_int("192.168.1.2")
        digits = ipv4_to_binary_digits("192.168.1.2")

        self.assertEqual(int_to_ipv4(value), "192.168.1.2")
        self.assertEqual(binary_digits_to_int(digits), value)
        self.assertEqual(digits.shape, (32,))

    def test_digits_are_network_prefix_first(self) -> None:
        digits = ipv4_to_binary_digits("192.168.1.2")

        self.assertEqual(digits[:8].tolist(), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(digits[8:16].tolist(), [1, 0, 1, 0, 1, 0, 0, 0])

    def test_invalid_ipv4_raises(self) -> None:
        with self.assertRaises(ValueError):
            ipv4_to_int("999.1.2.3")


class TestIPPrefixAnomalyDataset(unittest.TestCase):
    def _cfg(self, **kwargs) -> IPPrefixDatasetConfig:
        params = {
            "window_size": 8,
            "attack_fraction": 0.25,
            "prefix_len": 24,
            "num_prefixes": 4,
            "attack_min_len": 1,
            "attack_max_len": 2,
            "seed": 123,
        }
        params.update(kwargs)
        return IPPrefixDatasetConfig(**params)

    def test_dataset_shapes_and_label_balance(self) -> None:
        ds = IPPrefixAnomalyDataset(self._cfg(), n_samples=40)

        self.assertEqual(len(ds), 40)
        self.assertEqual(ds.windows.shape, (40, 8, 32))
        self.assertEqual(ds.window_prefix_ids.shape, (40, 8))
        self.assertEqual(int(ds.labels.sum().item()), 10)
        self.assertTrue(torch.all((ds.windows == 0) | (ds.windows == 1)))
        window, label = ds[0]
        self.assertEqual(window.shape, (8, 32))
        self.assertIn(float(label), (0.0, 1.0))

    def test_normal_windows_stay_inside_one_prefix(self) -> None:
        ds = IPPrefixAnomalyDataset(self._cfg(), n_samples=40)
        normal_prefix_ids = ds.window_prefix_ids[ds.labels == 0]

        for row in normal_prefix_ids:
            self.assertEqual(torch.unique(row).numel(), 1)

    def test_attack_windows_include_prefix_jump(self) -> None:
        ds = IPPrefixAnomalyDataset(
            self._cfg(attack_fraction=0.5, attack_kinds=("prefix_jump",)),
            n_samples=20,
        )
        attack_prefix_ids = ds.window_prefix_ids[ds.labels == 1]

        self.assertGreater(attack_prefix_ids.shape[0], 0)
        self.assertTrue(any(torch.unique(row).numel() > 1 for row in attack_prefix_ids))
        self.assertEqual(ds.attack_kind_counts["prefix_jump"], 10)

    def test_spoofed_prefix_is_marked_unknown(self) -> None:
        ds = IPPrefixAnomalyDataset(
            self._cfg(attack_fraction=0.5, attack_kinds=("spoofed_prefix",)),
            n_samples=20,
        )
        attack_prefix_ids = ds.window_prefix_ids[ds.labels == 1]

        self.assertTrue(bool((attack_prefix_ids == -1).any().item()))
        self.assertEqual(ds.attack_kind_counts["spoofed_prefix"], 10)

    def test_config_rejects_bad_attack_length(self) -> None:
        cfg = self._cfg(attack_min_len=3, attack_max_len=2)

        with self.assertRaises(ValueError):
            cfg.validate()

    def test_spoofed_prefix_requires_unused_prefix(self) -> None:
        cfg = self._cfg(
            prefix_len=1,
            num_prefixes=2,
            attack_kinds=("spoofed_prefix",),
        )

        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()

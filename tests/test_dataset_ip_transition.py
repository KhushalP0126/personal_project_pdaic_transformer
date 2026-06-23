"""Tests for transition-based IPv4 prefix anomaly datasets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from padic_transformer.dataset_ip_transition import (  # noqa: E402
    IPPrefixTransitionAnomalyDataset,
    IPPrefixTransitionDatasetConfig,
)


class TestIPPrefixTransitionAnomalyDataset(unittest.TestCase):
    def _cfg(self, **kwargs) -> IPPrefixTransitionDatasetConfig:
        params = {
            "window_size": 8,
            "attack_fraction": 0.25,
            "prefix_len": 24,
            "num_prefixes": 8,
            "num_groups": 2,
            "seed": 123,
        }
        params.update(kwargs)
        return IPPrefixTransitionDatasetConfig(**params)

    def _transition_violations(self, ds: IPPrefixTransitionAnomalyDataset, row: torch.Tensor) -> int:
        group_lookup = {}
        position_lookup = {}
        for group_id, group in enumerate(ds.groups):
            for pos, prefix_id in enumerate(group):
                group_lookup[prefix_id] = group_id
                position_lookup[prefix_id] = pos

        violations = 0
        for left_t, right_t in zip(row[:-1], row[1:]):
            left = int(left_t.item())
            right = int(right_t.item())
            if group_lookup[left] != group_lookup[right]:
                violations += 1
                continue
            group_size = ds.group_size
            expected = (position_lookup[left] + 1) % group_size
            if position_lookup[right] != expected:
                violations += 1
        return violations

    def test_dataset_shapes_and_label_balance(self) -> None:
        ds = IPPrefixTransitionAnomalyDataset(self._cfg(), n_samples=40)

        self.assertEqual(len(ds), 40)
        self.assertEqual(ds.windows.shape, (40, 8, 32))
        self.assertEqual(ds.window_prefix_ids.shape, (40, 8))
        self.assertEqual(int(ds.labels.sum().item()), 10)
        self.assertTrue(torch.all((ds.windows == 0) | (ds.windows == 1)))
        window, label = ds[0]
        self.assertEqual(window.shape, (8, 32))
        self.assertIn(float(label), (0.0, 1.0))

    def test_normal_windows_follow_cycle_rule(self) -> None:
        ds = IPPrefixTransitionAnomalyDataset(self._cfg(), n_samples=40)
        normal_prefix_ids = ds.window_prefix_ids[ds.labels == 0]

        self.assertGreater(normal_prefix_ids.shape[0], 0)
        for row in normal_prefix_ids:
            self.assertEqual(self._transition_violations(ds, row), 0)

    def test_attack_windows_violate_cycle_rule(self) -> None:
        ds = IPPrefixTransitionAnomalyDataset(
            self._cfg(attack_fraction=0.5, attack_kinds=("illegal_transition",)),
            n_samples=20,
        )
        attack_prefix_ids = ds.window_prefix_ids[ds.labels == 1]

        self.assertGreater(attack_prefix_ids.shape[0], 0)
        self.assertTrue(all(self._transition_violations(ds, row) > 0 for row in attack_prefix_ids))
        self.assertEqual(ds.attack_kind_counts["illegal_transition"], 10)

    def test_cycle_reversal_and_skip_are_supported(self) -> None:
        ds = IPPrefixTransitionAnomalyDataset(
            self._cfg(attack_fraction=0.5, attack_kinds=("cycle_reversal", "skip_transition")),
            n_samples=40,
        )
        attack_prefix_ids = ds.window_prefix_ids[ds.labels == 1]

        self.assertTrue(all(self._transition_violations(ds, row) > 0 for row in attack_prefix_ids))
        self.assertEqual(sum(ds.attack_kind_counts.values()), 20)

    def test_config_rejects_bad_grouping(self) -> None:
        cfg = self._cfg(num_prefixes=10, num_groups=3)

        with self.assertRaises(ValueError):
            cfg.validate()

    def test_config_rejects_too_small_groups(self) -> None:
        cfg = self._cfg(num_prefixes=4, num_groups=2)

        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()

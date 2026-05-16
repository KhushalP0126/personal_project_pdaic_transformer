"""Tests for model, dataset, loss, and training loop."""

from __future__ import annotations

import unittest

import torch

from padic_transformer.config import BenchmarkConfig
from padic_transformer.dataset import AnomalyDatasetConfig, SyscallAnomalyDataset, build_dataloaders
from padic_transformer.losses import AnomalyLoss, PadicContrastiveLoss
from padic_transformer.model import HenselEmbedding, PadicAnomalyDetector
from padic_transformer.training import TrainConfig, train
from padic_transformer.ultrametric import generate_clustered_hensel_dataset


class TestHenselEmbedding(unittest.TestCase):
    def _make(self, p: int = 3, r: int = 8, d: int = 32) -> HenselEmbedding:
        return HenselEmbedding(p=p, r=r, d_model=d)

    def test_output_shape(self) -> None:
        emb = self._make()
        digits = torch.randint(0, 3, (4, 16, 8))
        out = emb(digits)
        self.assertEqual(out.shape, (4, 16, 32))

    def test_bad_p_raises(self) -> None:
        with self.assertRaises(ValueError):
            HenselEmbedding(p=1, r=8, d_model=32)

    def test_wrong_last_dim_raises(self) -> None:
        emb = self._make(r=8)
        with self.assertRaises(ValueError):
            emb(torch.zeros(2, 4, 5, dtype=torch.int64))


class TestPadicAnomalyDetector(unittest.TestCase):
    def _small_model(self) -> PadicAnomalyDetector:
        return PadicAnomalyDetector(p=3, r=8, d_model=32, n_heads=4, n_layers=1, ffn_dim=64, head_hidden=16)

    def test_forward_shape(self) -> None:
        model = self._small_model()
        digits = torch.randint(0, 3, (8, 16, 8))
        logits = model(digits)
        self.assertEqual(logits.shape, (8,))

    def test_parameter_summary_runs(self) -> None:
        model = self._small_model()
        s = model.parameter_summary()
        self.assertIn("Total", s)
        self.assertGreater(model.count_parameters(), 0)

    def test_padding_mask_accepted(self) -> None:
        model = self._small_model()
        digits = torch.randint(0, 3, (4, 10, 8))
        mask = torch.zeros(4, 10, dtype=torch.bool)
        mask[:, -2:] = True
        logits = model(digits, padding_mask=mask)
        self.assertEqual(logits.shape, (4,))


class TestSyscallAnomalyDataset(unittest.TestCase):
    def _make_hensel(self, seed: int = 42) -> object:
        cfg = BenchmarkConfig(p=3, r=8, samples=256, classes=4, tokens_per_class=64, seed=seed)
        return generate_clustered_hensel_dataset(cfg)

    def test_dataset_lengths(self) -> None:
        hensel = self._make_hensel()
        acfg = AnomalyDatasetConfig(window_size=8, attack_fraction=0.4, attack_min_len=1, attack_max_len=3, seed=0)
        ds = SyscallAnomalyDataset(hensel, acfg, n_samples=100)
        self.assertEqual(len(ds), 100)

    def test_item_shapes(self) -> None:
        hensel = self._make_hensel()
        acfg = AnomalyDatasetConfig(window_size=8, attack_fraction=0.3, attack_min_len=1, attack_max_len=4, seed=1)
        ds = SyscallAnomalyDataset(hensel, acfg, n_samples=50)
        window, label = ds[0]
        self.assertEqual(window.shape, (8, 8))
        self.assertIn(float(label), (0.0, 1.0))

    def test_attack_fraction_approximate(self) -> None:
        hensel = self._make_hensel()
        acfg = AnomalyDatasetConfig(window_size=8, attack_fraction=0.5, attack_min_len=1, attack_max_len=4, seed=2)
        ds = SyscallAnomalyDataset(hensel, acfg, n_samples=200)
        n_attack = int(ds.labels.sum().item())
        self.assertAlmostEqual(n_attack / 200, 0.5, delta=0.05)

    def test_build_dataloaders_runs(self) -> None:
        bcfg = BenchmarkConfig(p=3, r=8, samples=256, classes=4, tokens_per_class=32, seed=0)
        acfg = AnomalyDatasetConfig(window_size=8, attack_fraction=0.3, attack_min_len=1, attack_max_len=4, seed=0)
        device = torch.device("cpu")
        train_dl, val_dl = build_dataloaders(bcfg, acfg, n_train=64, n_val=16, batch_size=16, device=device, num_workers=0)
        batch = next(iter(train_dl))
        windows, labels = batch
        self.assertEqual(windows.shape, (16, 8, 8))
        self.assertEqual(labels.shape, (16,))


class TestLosses(unittest.TestCase):
    def test_contrastive_loss_positive(self) -> None:
        digits = torch.randint(0, 3, (8, 4, 8))
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
        loss_fn = PadicContrastiveLoss(p=3, margin_pos=0.1, margin_neg=0.5)
        loss = loss_fn(digits, labels)
        self.assertGreaterEqual(float(loss.item()), 0.0)
        self.assertFalse(torch.isnan(loss))

    def test_anomaly_loss_returns_three_scalars(self) -> None:
        digits = torch.randint(0, 3, (8, 4, 8))
        labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
        logits = torch.randn(8)
        loss_fn = AnomalyLoss(p=3, alpha=0.5)
        total, bce, ctr = loss_fn(logits, labels, digits)
        for t in (total, bce, ctr):
            self.assertEqual(t.shape, ())
            self.assertFalse(torch.isnan(t))

    def test_zero_alpha_equals_pure_bce(self) -> None:
        digits = torch.randint(0, 3, (8, 4, 8))
        labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
        logits = torch.randn(8)
        loss_fn = AnomalyLoss(p=3, alpha=0.0)
        total, bce, _ = loss_fn(logits, labels, digits)
        torch.testing.assert_close(total, bce, rtol=0, atol=1e-6)


class TestTrainingSmoke(unittest.TestCase):
    def test_one_epoch_cpu(self) -> None:
        cfg = TrainConfig(
            p=3,
            r=8,
            d_model=32,
            n_heads=4,
            n_layers=1,
            ffn_dim=64,
            head_hidden=16,
            window_size=8,
            n_train=64,
            n_val=16,
            samples=256,
            classes=4,
            tokens_per_class=32,
            epochs=1,
            batch_size=16,
            num_workers=0,
            checkpoint_dir="results/test_checkpoints",
            log_json="results/test_training_log.json",
            log_md="results/test_training_log.md",
            save_every=999,
        )
        result = train(cfg, torch.device("cpu"))
        self.assertIn("best_auroc", result)
        self.assertGreaterEqual(result["best_auroc"], 0.0)
        self.assertLessEqual(result["best_auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()

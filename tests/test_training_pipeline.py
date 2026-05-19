"""Tests for model, dataset, loss, and training loop."""

from __future__ import annotations

import unittest
from unittest.mock import patch
import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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

    def test_forward_with_features_has_gradient_path_shape(self) -> None:
        model = self._small_model()
        digits = torch.randint(0, 3, (4, 10, 8))
        logits, features = model.forward_with_features(digits)
        self.assertEqual(logits.shape, (4,))
        self.assertEqual(features.shape, (4, 32))


class TestSyscallAnomalyDataset(unittest.TestCase):
    def _make_hensel(self, seed: int = 42) -> object:
        cfg = BenchmarkConfig(p=3, r=8, samples=256, classes=4, tokens_per_class=64, seed=seed)
        return generate_clustered_hensel_dataset(cfg)

    def test_dataset_lengths(self) -> None:
        hensel = self._make_hensel()
        acfg = AnomalyDatasetConfig(window_size=8, attack_fraction=0.4, attack_min_len=1, attack_max_len=3, seed=0)
        ds = SyscallAnomalyDataset(hensel, acfg, n_samples=100)
        self.assertEqual(len(ds), 100)

    def test_stream_is_not_single_contiguous_class_blocks(self) -> None:
        hensel = self._make_hensel(seed=7)
        changes = (hensel.token_labels[1:] != hensel.token_labels[:-1]).sum().item()
        self.assertGreater(changes, 0)

    def test_item_shapes(self) -> None:
        hensel = self._make_hensel()
        acfg = AnomalyDatasetConfig(window_size=8, attack_fraction=0.3, attack_min_len=1, attack_max_len=4, seed=1)
        ds = SyscallAnomalyDataset(hensel, acfg, n_samples=50)
        window, label = ds[0]
        self.assertEqual(window.shape, (8, 8))
        self.assertIn(float(label), (0.0, 1.0))

    def test_small_sample_rejects_empty_split(self) -> None:
        hensel = self._make_hensel()
        acfg = AnomalyDatasetConfig(window_size=8, attack_fraction=0.5, attack_min_len=1, attack_max_len=4, seed=3)
        with self.assertRaises(ValueError):
            SyscallAnomalyDataset(hensel, acfg, n_samples=1)

    def test_inject_attack_uses_actual_injection_region(self) -> None:
        from padic_transformer.dataset import SyscallAnomalyDataset

        tokens = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
            ],
            dtype=torch.int64,
        )
        labels = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.int64)
        hensel = type("DummyHensel", (), {"token_digits": tokens, "token_labels": labels})()
        ds = SyscallAnomalyDataset.__new__(SyscallAnomalyDataset)
        ds.hensel_data = hensel
        ds.cfg = AnomalyDatasetConfig(window_size=4, attack_fraction=0.5, attack_min_len=1, attack_max_len=1, seed=0)

        randint_values = iter(
            [
                torch.tensor([1], dtype=torch.int64),
                torch.tensor([2], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
            ]
        )

        def fake_randint(*args, **kwargs):
            return next(randint_values)

        with patch("torch.randint", side_effect=fake_randint):
            attacked = ds._inject_attack(start=0, window=4, rng=torch.Generator())

        self.assertFalse(torch.equal(attacked[2], tokens[2]))
        matches = (attacked[2] == tokens).all(dim=1).nonzero(as_tuple=True)[0]
        self.assertGreater(matches.numel(), 0)
        self.assertNotEqual(int(labels[2].item()), int(labels[matches[0]].item()))

    def test_get_window_does_not_wrap(self) -> None:
        hensel = self._make_hensel()
        acfg = AnomalyDatasetConfig(window_size=8, attack_fraction=0.3, attack_min_len=1, attack_max_len=4, seed=1)
        ds = SyscallAnomalyDataset(hensel, acfg, n_samples=20)
        with self.assertRaises(ValueError):
            ds._get_window(hensel.token_digits.shape[0] - 4, 8)

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
        reps = torch.randn(8, 16)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
        loss_fn = PadicContrastiveLoss(p=3, margin_pos=0.1, margin_neg=0.5)
        loss = loss_fn(reps, labels)
        self.assertGreaterEqual(float(loss.item()), 0.0)
        self.assertFalse(torch.isnan(loss))

    def test_anomaly_loss_returns_three_scalars(self) -> None:
        reps = torch.randn(8, 16, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
        logits = torch.randn(8)
        loss_fn = AnomalyLoss(p=3, alpha=0.5)
        total, bce, ctr = loss_fn(logits, labels, reps)
        for t in (total, bce, ctr):
            self.assertEqual(t.shape, ())
            self.assertFalse(torch.isnan(t))
        total.backward()
        self.assertIsNotNone(reps.grad)

    def test_zero_alpha_equals_pure_bce(self) -> None:
        reps = torch.randn(8, 16)
        labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
        logits = torch.randn(8)
        loss_fn = AnomalyLoss(p=3, alpha=0.0)
        total, bce, _ = loss_fn(logits, labels, reps)
        torch.testing.assert_close(total, bce, rtol=0, atol=1e-6)

    def test_batch_one_returns_zero(self) -> None:
        reps = torch.randn(1, 16)
        labels = torch.tensor([0.0], dtype=torch.float32)
        loss_fn = PadicContrastiveLoss(p=3, margin_pos=0.1, margin_neg=0.5)
        loss = loss_fn(reps, labels)
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.item()), 0.0)


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

    def test_grad_accum_validation(self) -> None:
        from padic_transformer.training import _train_epoch

        with self.assertRaises(ValueError):
            _train_epoch(
                model=torch.nn.Linear(1, 1),
                loader=[],
                loss_fn=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1),
                device=torch.device("cpu"),
                amp_dtype=torch.float32,
                max_grad_norm=1.0,
                grad_accum=0,
                scaler=None,
            )

    def test_threshold_search_metrics_reports_best_f1(self) -> None:
        from padic_transformer.training import _search_threshold_metrics

        logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        metrics = _search_threshold_metrics(logits, labels, num_thresholds=16)
        self.assertIn("threshold", metrics)
        self.assertGreaterEqual(metrics["f1"], 0.0)
        self.assertGreaterEqual(metrics["precision"], 0.0)
        self.assertGreaterEqual(metrics["recall"], 0.0)
        self.assertGreaterEqual(metrics["fpr"], 0.0)

    def test_cuda_optimizer_fallback_path_is_supported(self) -> None:
        import torch.optim
        from padic_transformer import training as training_module

        class DummyAdamW(torch.optim.AdamW):
            def __init__(self, *args, **kwargs):
                if kwargs.get("fused"):
                    raise RuntimeError("fused not supported")
                super().__init__(*args, **kwargs)

        with patch.object(training_module.torch.optim, "AdamW", DummyAdamW):
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
            result = training_module.train(cfg, torch.device("cpu"))
            self.assertIn("best_auroc", result)

    def test_padic_distance_loss_empty_mask_is_finite(self) -> None:
        from scripts.experiment_controller import padic_distance_loss

        digits = torch.tensor([[0, 1, 0], [0, 1, 1]], dtype=torch.int64)
        pairs = torch.tensor([[0, 1]], dtype=torch.int64)
        all_close = torch.tensor([True], dtype=torch.bool)
        all_far = torch.tensor([False], dtype=torch.bool)

        loss_close = padic_distance_loss(digits, pairs, all_close)
        loss_far = padic_distance_loss(digits, pairs, all_far)
        self.assertTrue(torch.isfinite(loss_close))
        self.assertTrue(torch.isfinite(loss_far))

    def test_over_underfit_classifier(self) -> None:
        from scripts.over_underfit import classify

        history = [
            {"train": {"loss": 1.0}, "val": {"loss": 1.0, "best_f1": 0.40}},
            {"train": {"loss": 0.6}, "val": {"loss": 0.9, "best_f1": 0.55}},
        ]
        self.assertEqual(classify(history), "learning/generalizing")

        history_overfit = [
            {"train": {"loss": 1.0}, "val": {"loss": 1.0, "best_f1": 0.40}},
            {"train": {"loss": 0.5}, "val": {"loss": 1.3, "best_f1": 0.42}},
        ]
        self.assertEqual(classify(history_overfit), "likely overfitting")


if __name__ == "__main__":
    unittest.main()

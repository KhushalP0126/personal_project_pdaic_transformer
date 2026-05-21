# Project Context

## Goal

Build a p-adic transformer research harness for the Privacy Chip project. The first milestone is an organized synthetic data generator and benchmark that can be run locally or on a cloud GPU before hardware-specific RISCA/DE2-115 work.

## p-adic Configuration

- Baseline prime: `p = 3`
- Comparison prime: `p = 5`
- Local reference precision: `r in {8, 16}`
- Cloud GPU sweep precision: `r in {8, 16, 24, 32}`

Hensel codes are represented as fixed-width digit tensors with shape:

```text
[items, r]
```

Digit index `0` is the least significant p-adic digit. Two tokens are p-adically close when they share a longer low-order Hensel prefix.

## Ultrametric Loss Axiom

For generated tokens `x`, `y`, and `z`, the p-adic distance must satisfy:

```text
d(x, z) <= max(d(x, y), d(y, z))
```

The implementation validates this using integer valuations instead of floating-point distances. If `v(a, b)` is the number of shared low-order Hensel digits, the inequality is checked as:

```text
v(x, z) >= min(v(x, y), v(y, z))
```

## Security And Reproducibility

- No dynamic `eval` or remote code loading.
- Deterministic RNG seeds for benchmarks.
- CLI sizes are bounded to prevent accidental memory explosions.
- Generated results are written only to the local `results/` directory.

## Progress Snapshot

- Overall completion: about `42%`.
- Core infrastructure: about `85%`.
- Training pipeline: about `75%`.
- Model architecture: about `55%`.
- Evaluation and validation: about `30%`.
- Deployment readiness: about `10%`.

The remaining work is mostly evidence and integration work rather than basic plumbing. The current synthetic pipeline and CPU training path work, but real-data baselines, ablations, threshold calibration, multi-seed stability, and deployment-oriented inference remain open.

## Session Log

- Opened and inspected `claude_attention.zip`, then implemented the soft p-adic attention module in `src/padic_transformer/padic_attention.py`.
- Exported the new attention symbols from `src/padic_transformer/__init__.py`.
- Made the training loop model-agnostic so the attention model can be selected without changing the synthetic dataset pipeline.
- Added `--attention` and `--d-digit` support to `scripts/train_anomaly_detector.py`.
- Added a dedicated attention smoke test file in `tests/test_padic_attention.py`.
- Wired the attention model into `Makefile` with CPU and GPU targets.
- Installed and verified `torch` in the local venv, then ran the CPU attention training path successfully.
- Removed stale archive and generated artifacts when cleaning the workspace.
- Fixed the synthetic stream pipeline so it behaves like a sequential watchdog signal rather than contiguous class blocks.
- Fixed the top pipeline bugs: latent-feature contrastive loss, threshold scale mismatch, non-wrapping windows, seed separation, warning paths, AMP validation parity, metric aggregation, and newer `GradScaler` usage.
- Fixed the attention implementation details: batched logits, temperature clamp, padded-query masking, and explicit contiguous reshaping.
- Updated the experiment controller smoke test so anomaly F1 is measured as class `1`, and added a `p=3` bit-flip noise check.
- Added `validate=True` support to `int64_to_digits()` so truncation can be checked explicitly.
- Switched back to `main`, pulled the latest `origin/main`, and pushed the reshaping fix to `main`.

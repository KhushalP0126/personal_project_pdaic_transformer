# Reproducibility

## Environment

```bash
make setup
```

## Validation

```bash
make test
make check-study
```

If you are already set up and want the offline-safe test path without a reinstall
step:

```bash
make test-unit
```

## Main study

```bash
make ip-study-cpu
```

Outputs:

```text
results/final_summary.json
results/final_summary.md
results/final_summary_by_seed.md
```

## What the study runs

Variants:

- `standard_transformer`
- `flat_digit_transformer`
- `hensel_only`
- `hensel_padic_sigmoid`
- `hensel_padic_signed_alpha`

Tasks:

- simple synthetic IP-prefix anomalies
- transition-rule synthetic IP anomalies
- cross-generator simple->transition
- cross-generator transition->simple
- realistic idle-heavy proxy

## Hardware and runtime note

All final study numbers were run on CPU with:

- seeds: `20260504`, `20260505`, `20260506`
- epochs: `3`
- `d_model=64`
- `n_layers=1`
- `n_heads=4`
- `train_samples=2048`
- `val_samples=512`

The realistic proxy uses the same train/validation counts with:

- `realistic_window_size=32`
- `realistic_attack_fraction=0.05`
- `realistic_idle_fraction=0.70`

## Current result snapshot

From `results/final_summary.md`:

- Simple synthetic: signed alpha `0.6065 +- 0.0191`, old gate `0.5811 +- 0.0461`, hensel-only `0.5703 +- 0.0634`, flat digit `0.5372 +- 0.0209`, standard `0.5000`
- Transition synthetic: standard `0.5000` is best; signed alpha is `0.4982 +- 0.0075`
- Cross simple->transition: signed alpha `0.5016 +- 0.0257`, old gate `0.5003 +- 0.0402`, standard `0.5000`
- Cross transition->simple: flat digit `0.5148 +- 0.0467` is best
- Realistic proxy: standard `0.7446 +- 0.0530`, flat digit `0.7315 +- 0.0741`, hensel-only `0.7186 +- 0.0657`, signed alpha `0.7100 +- 0.0582`, old gate `0.6926 +- 0.0488`

Interpretation:

- Simple synthetic still shows a real structured-signal win.
- Signed alpha beats the old gate on the simple task by keeping alpha near zero, not by learning a strong positive p-adic pull.
- Transition and generator-shift rows are close to chance, so transfer remains weak.
- The flat digit baseline is strong enough that the paper must separate “digitized prefix structure helps” from “specifically Hensel structure helps.”
- The old sigmoid attention bias is not robust.
- Transfer remains weak under generator shift.

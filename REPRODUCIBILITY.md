# Reproducibility

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

## Validation

```bash
make test
```

Targeted study checks:

```bash
.venv/bin/python -m unittest tests.test_padic_attention tests.test_training_pipeline
.venv/bin/python -m py_compile scripts/run_ip_characterization_study.py
```

## Main study

```bash
make ip-study-cpu
```

Equivalent direct command:

```bash
.venv/bin/python scripts/run_ip_characterization_study.py \
  --device cpu \
  --seeds 20260504 20260505 20260506
```

Outputs:

```text
results/final_summary.json
results/final_summary.md
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

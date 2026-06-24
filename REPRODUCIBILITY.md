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
.venv/bin/python scripts/run_ip_characterization_study.py --device cpu
```

Outputs:

```text
results/final_summary.json
results/final_summary.md
```

## What the study runs

Variants:

- `standard_transformer`
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

- Simple synthetic: signed alpha `0.6260`, old gate `0.6089`, hensel-only `0.5846`, standard `0.5000`
- Transition synthetic: signed alpha `0.5252`, old gate `0.5205`, hensel-only `0.4622`, standard `0.5000`
- Cross simple->transition: hensel-only `0.5400` is best
- Cross transition->simple: standard `0.5000` is best, all others are worse
- Realistic proxy: standard `0.7485`, signed alpha `0.7484`, hensel-only `0.6895`, old gate `0.6013`

Interpretation:

- Hensel coordinates matter on the IP tasks.
- The old sigmoid attention bias is not robust.
- Signed alpha often matches or beats the old gate by shrinking the explicit bias toward zero.
- Transfer remains weak under generator shift.

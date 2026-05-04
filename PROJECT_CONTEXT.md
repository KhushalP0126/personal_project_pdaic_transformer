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


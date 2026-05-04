# Personal Project: p-adic Transformer

Phase 1 creates a reproducible PyTorch p-adic benchmark harness for the Privacy Chip workflow:

- generate synthetic Hensel-code token clusters,
- validate the ultrametric triangle inequality,
- compare `p=3` and `p=5` at configurable precision `r`,
- time p-adic conversion and distance kernels,
- run on CUDA when PyTorch with GPU support is available.

## Quick Start

```bash
python3 -m pip install -e .
python3 scripts/run_padic_benchmark.py \
  --p-list 3 5 \
  --r-list 8 16 \
  --samples 4096 \
  --classes 16 \
  --tokens-per-class 256
```

For a cloud GPU sweep:

```bash
python3 -m pip install -e .
python3 scripts/run_padic_benchmark.py \
  --device cuda \
  --p-list 3 5 \
  --r-list 8 16 24 32 \
  --samples 16384 \
  --classes 32 \
  --tokens-per-class 128
```

## Validation

Run the standard-library tests:

```bash
python3 -m unittest discover -s tests
```

The local reference output is stored in [`results/reference_benchmark.md`](results/reference_benchmark.md).

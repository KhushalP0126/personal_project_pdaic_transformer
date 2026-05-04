# Personal Project: p-adic Transformer

Phase 1 creates a reproducible PyTorch p-adic benchmark harness for the Privacy Chip workflow:

- generate synthetic Hensel-code token clusters,
- validate the ultrametric triangle inequality,
- compare `p=3` and `p=5` at configurable precision `r`,
- time p-adic conversion and distance kernels,
- run on CUDA when PyTorch with GPU support is available.

## Quick Start

```bash
make
```

For a cloud GPU sweep:

```bash
make gpu
```

For the INT8 2-adic hardware dry-lab:

```bash
make int8
```

The benchmark arguments are configurable:

```bash
make cpu CPU_ARGS="--device cpu --p-list 3 5 --r-list 16 --samples 8192"
make gpu GPU_ARGS="--device cuda --p-list 3 5 --r-list 8 16 24 32 --samples 32768"
```

## Validation

Run the standard-library tests:

```bash
make test
```

The local reference output is stored in [`results/reference_benchmark.md`](results/reference_benchmark.md).

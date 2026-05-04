# Reference p-adic Benchmark

- Generated UTC: `2026-05-04T17:58:05Z`
- Python: `3.11.9`
- Platform: `macOS-26.2-arm64-arm-64bit`
- Torch: `2.11.0`
- CUDA available: `False`

| p | r | nearest-center accuracy | violations | violation rate | backend | device | distance pairs/s | int64 pack |
|---:|---:|---:|---:|---:|---|---|---:|---|
| 3 | 8 | 0.8818 | 0 | 0.000000 | torch | cpu | 17474689.03 | 0.000443s |
| 3 | 16 | 1.0000 | 0 | 0.000000 | torch | cpu | 7192142.58 | 0.000602s |
| 5 | 8 | 1.0000 | 0 | 0.000000 | torch | cpu | 34330832.49 | 0.000277s |
| 5 | 16 | 1.0000 | 0 | 0.000000 | torch | cpu | 23125507.14 | 0.000470s |

Ultrametric validation uses integer p-adic valuations, not floating-point distance comparisons.
The cloud GPU path uses the same PyTorch benchmark script with `--device cuda`.

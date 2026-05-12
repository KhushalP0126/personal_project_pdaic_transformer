# Reference p-adic Benchmark

- Generated UTC: `2026-05-04T18:11:22Z`
- Python: `3.11.9`
- Platform: `macOS-26.2-arm64-arm-64bit`
- Torch: `2.11.0`
- CUDA available: `False`

| p | r | nearest-center accuracy | violations | violation rate | backend | device | distance pairs/s | int64 pack |
|---:|---:|---:|---:|---:|---|---|---:|---|
| 3 | 8 | 0.8818 | 0 | 0.000000 | torch | cpu | 23458003.03 | 0.012698s |
| 3 | 16 | 1.0000 | 0 | 0.000000 | torch | cpu | 8121909.86 | 0.000609s |
| 5 | 8 | 1.0000 | 0 | 0.000000 | torch | cpu | 28053768.06 | 0.000291s |
| 5 | 16 | 1.0000 | 0 | 0.000000 | torch | cpu | 22572089.75 | 0.000483s |

Ultrametric validation uses integer p-adic valuations, not floating-point distance comparisons.
The cloud GPU path uses the same PyTorch benchmark script with `--device cuda`.

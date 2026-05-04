# Reference p-adic Benchmark

- Generated UTC: `2026-05-04T17:27:13Z`
- Python: `3.11.9`
- Platform: `macOS-26.2-arm64-arm-64bit`
- Torch: `2.11.0`
- CUDA available: `False`

| p | r | nearest-center accuracy | violations | violation rate | backend | device | distance pairs/s | int64 pack |
|---:|---:|---:|---:|---:|---|---|---:|---|
| 3 | 8 | 0.8818 | 0 | 0.000000 | torch | cpu | 16817145.06 | 0.000463s |
| 3 | 16 | 1.0000 | 0 | 0.000000 | torch | cpu | 7516218.57 | 0.000595s |
| 5 | 8 | 1.0000 | 0 | 0.000000 | torch | cpu | 36098639.43 | 0.000292s |
| 5 | 16 | 1.0000 | 0 | 0.000000 | torch | cpu | 22458031.41 | 0.000473s |

Ultrametric validation uses integer p-adic valuations, not floating-point distance comparisons.
The cloud GPU path uses the same PyTorch benchmark script with `--device cuda`.

# Reference p-adic Benchmark

- Generated UTC: `2026-05-21T15:33:46Z`
- Python: `3.11.9`
- Platform: `macOS-26.2-arm64-arm-64bit`
- Torch: `2.11.0`
- CUDA available: `False`

| p | r | nearest-center accuracy | violations | violation rate | backend | device | distance pairs/s | int64 pack |
|---:|---:|---:|---:|---:|---|---|---:|---|
| 3 | 8 | 0.8750 | 0 | 0.000000 | torch | cpu | 23732771.14 | 0.004171s |
| 3 | 16 | 1.0000 | 0 | 0.000000 | torch | cpu | 8922165.01 | 0.000439s |
| 5 | 8 | 1.0000 | 0 | 0.000000 | torch | cpu | 23747446.64 | 0.000270s |
| 5 | 16 | 1.0000 | 0 | 0.000000 | torch | cpu | 22997427.14 | 0.000420s |

Ultrametric validation uses integer p-adic valuations, not floating-point distance comparisons.
The cloud GPU path uses the same PyTorch benchmark script with `--device cuda`.

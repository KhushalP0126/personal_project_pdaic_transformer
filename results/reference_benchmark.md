# Reference p-adic Benchmark

- Generated UTC: `2026-05-24T01:08:49Z`
- Python: `3.12.4`
- Platform: `Linux-5.14.0-362.8.1.el9_3.x86_64-x86_64-with-glibc2.34`
- Torch: `2.6.0+cu124`
- CUDA available: `True`

| p | r | nearest-center accuracy | violations | violation rate | backend | device | distance pairs/s | int64 pack |
|---:|---:|---:|---:|---:|---|---|---:|---|
| 3 | 8 | 0.7841 | 0 | 0.000000 | torch | cuda | 129900108.20 | 0.009119s |
| 3 | 16 | 1.0000 | 0 | 0.000000 | torch | cuda | 126959298.28 | 0.000269s |
| 3 | 24 | 1.0000 | 0 | 0.000000 | torch | cuda | 124130804.52 | 0.000296s |
| 3 | 32 | 1.0000 | 0 | 0.000000 | torch | cuda | 120673069.67 | 0.000360s |
| 5 | 8 | 1.0000 | 0 | 0.000000 | torch | cuda | 132455555.84 | 0.000159s |
| 5 | 16 | 1.0000 | 0 | 0.000000 | torch | cuda | 128869367.35 | 0.000223s |
| 5 | 24 | 1.0000 | 0 | 0.000000 | torch | cuda | 126477971.76 | 0.000287s |
| 5 | 32 | 1.0000 | 0 | 0.000000 | torch | cuda | 122472744.86 | skipped |

Ultrametric validation uses integer p-adic valuations, not floating-point distance comparisons.
The cloud GPU path uses the same PyTorch benchmark script with `--device cuda`.

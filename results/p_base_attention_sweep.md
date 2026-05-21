# p-Basis Attention Sweep

- Generated UTC: `2026-05-21T01:07:42Z`
- Device: `cpu`
- Window size: `32`
- Batches per p: `4`
- Batch size: `64`

| p | Attention Sparsity | normal d_p | anomaly d_p | forward latency |
|---:|---:|---:|---:|---:|
| 2 | 0.0000% | 0.604129 | 0.617128 | 116.793 |
| 3 | 0.0000% | 0.630680 | 0.642066 | 123.669 |
| 5 | 0.0000% | 0.763647 | 0.767418 | 127.594 |
| 7 | 0.0000% | 0.770347 | 0.775249 | 131.061 |

Attention sparsity is the percentage of attention weights below `1e-4`.
The p-adic distance uses `d_p(x, y) = p^{-v_p(x-y)}` averaged over non-diagonal token pairs.

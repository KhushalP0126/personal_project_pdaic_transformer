# p-Basis Attention Sweep

- Generated UTC: `2026-05-24T03:26:19Z`
- Device: `cpu`
- Window size: `32`
- Batches per p: `4`
- Batch size: `64`

| p | Attention Sparsity | hierarchy corr | hierarchy gap | p-adic gate | normal d_p | anomaly d_p | forward latency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.0000% | 0.6335 | 0.005844 | 0.1192 | 0.609743 | 0.620951 | 813.153 |
| 3 | 0.0000% | 0.4187 | 0.005056 | 0.1192 | 0.626469 | 0.636659 | 784.623 |
| 5 | 0.0000% | 0.3380 | 0.006050 | 0.1192 | 0.761987 | 0.769488 | 1499.209 |
| 7 | 0.0000% | 0.3411 | 0.006249 | 0.1192 | 0.764151 | 0.776083 | 911.896 |

Attention sparsity is the percentage of attention weights below `1e-4`.
Hierarchy correlation is the correlation between attention weights and hard shared-prefix length on non-diagonal valid token pairs.
The p-adic distance uses `d_p(x, y) = p^{-v_p(x-y)}` averaged over non-diagonal token pairs.

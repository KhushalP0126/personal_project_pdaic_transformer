# IP Synthetic Experiment

## Dataset
- Train samples: `2048`
- Val samples: `512`
- Window size: `16`
- Prefix length: `/24`
- Train positive rate: `0.2998`
- Val positive rate: `0.2988`

## Results

| Model | AUROC | F1 | Precision | Recall | p-adic corr | hierarchy gap | gate | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.5260 | 0.3810 | - | - | - | - | - | 0.08 |
| isolation_forest | 0.5098 | 0.4602 | 0.2988 | 1.0000 | - | - | - | 0.15 |
| vanilla_transformer | 0.4568 | 0.1840 | 0.2371 | 0.1503 | - | - | - | 1.50 |
| padic_attention_true | 0.5062 | 0.3290 | 0.3248 | 0.3333 | 0.3005 | 0.0311 | 0.4997 | 28.33 |
| padic_attention_shuffled | 0.4749 | 0.4267 | 0.2979 | 0.7516 | 0.2111 | 0.0271 | 0.5002 | 31.08 |
| padic_attention_random | 0.4929 | 0.3909 | 0.2997 | 0.5621 | 0.2645 | 0.0284 | 0.5003 | 37.37 |

## Notes

- `padic_attention_true` keeps MSB-first IP prefix bits.
- `padic_attention_shuffled` randomly permutes the unique IP-token vocabulary.
- `padic_attention_random` remaps each unique IP token to random 32-bit digits.

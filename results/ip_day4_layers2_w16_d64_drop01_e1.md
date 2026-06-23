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
| logistic_regression | 0.5503 | 0.4140 | - | - | - | - | - | 0.02 |
| isolation_forest | 0.5452 | 0.4602 | 0.2988 | 1.0000 | - | - | - | 0.14 |
| vanilla_transformer | 0.4816 | 0.2443 | 0.2936 | 0.2092 | - | - | - | 0.83 |
| padic_attention_true | 0.5723 | 0.4288 | 0.2990 | 0.7582 | 0.0717 | 0.0188 | 0.4996 | 21.99 |
| padic_attention_shuffled | 0.5062 | 0.0000 | 0.0000 | 0.0000 | 0.2041 | 0.0332 | 0.4998 | 22.75 |
| padic_attention_random | 0.5398 | 0.4414 | 0.3148 | 0.7386 | 0.2105 | 0.0329 | 0.5001 | 23.26 |

## Notes

- `padic_attention_true` keeps MSB-first IP prefix bits.
- `padic_attention_shuffled` randomly permutes the unique IP-token vocabulary.
- `padic_attention_random` remaps each unique IP token to random 32-bit digits.

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
| isolation_forest | 0.5452 | 0.4602 | 0.2988 | 1.0000 | - | - | - | 0.15 |
| vanilla_transformer | 0.4620 | 0.2458 | 0.2500 | 0.2418 | - | - | - | 0.51 |
| padic_attention_true | 0.5723 | 0.4654 | 0.3064 | 0.9673 | 0.0886 | 0.0240 | 0.5000 | 10.15 |
| padic_attention_shuffled | 0.4671 | 0.1975 | 0.2667 | 0.1569 | 0.1951 | 0.0327 | 0.5000 | 11.02 |
| padic_attention_random | 0.4639 | 0.4602 | 0.2988 | 1.0000 | 0.2114 | 0.0321 | 0.5000 | 16.58 |

## Notes

- `padic_attention_true` keeps MSB-first IP prefix bits.
- `padic_attention_shuffled` randomly permutes the unique IP-token vocabulary.
- `padic_attention_random` remaps each unique IP token to random 32-bit digits.

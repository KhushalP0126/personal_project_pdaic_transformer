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
| logistic_regression | 0.5570 | 0.4111 | - | - | - | - | - | 0.04 |
| isolation_forest | 0.5452 | 0.4602 | 0.2988 | 1.0000 | - | - | - | 0.15 |
| vanilla_transformer | 0.5339 | 0.4415 | 0.3103 | 0.7647 | - | - | - | 1.43 |
| padic_attention_true | 0.6559 | 0.4963 | 0.3463 | 0.8758 | 0.0939 | 0.0238 | 0.4986 | 31.67 |
| padic_attention_shuffled | 0.5111 | 0.0000 | 0.0000 | 0.0000 | 0.2183 | 0.0319 | 0.5000 | 31.08 |
| padic_attention_random | 0.5352 | 0.4668 | 0.3057 | 0.9869 | 0.2031 | 0.0270 | 0.4999 | 30.74 |

## Notes

- `padic_attention_true` keeps MSB-first IP prefix bits.
- `padic_attention_shuffled` randomly permutes the unique IP-token vocabulary.
- `padic_attention_random` remaps each unique IP token to random 32-bit digits.

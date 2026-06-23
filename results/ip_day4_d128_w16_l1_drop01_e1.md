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
| isolation_forest | 0.5452 | 0.4602 | 0.2988 | 1.0000 | - | - | - | 0.19 |
| vanilla_transformer | 0.5271 | 0.3189 | 0.3243 | 0.3137 | - | - | - | 1.02 |
| padic_attention_true | 0.5301 | 0.2128 | 0.3049 | 0.1634 | 0.0859 | 0.0215 | 0.4997 | 15.35 |
| padic_attention_shuffled | 0.5432 | 0.4602 | 0.2988 | 1.0000 | 0.1750 | 0.0314 | 0.4999 | 15.47 |
| padic_attention_random | 0.5145 | 0.4602 | 0.2988 | 1.0000 | 0.2016 | 0.0305 | 0.4998 | 11.40 |

## Notes

- `padic_attention_true` keeps MSB-first IP prefix bits.
- `padic_attention_shuffled` randomly permutes the unique IP-token vocabulary.
- `padic_attention_random` remaps each unique IP token to random 32-bit digits.

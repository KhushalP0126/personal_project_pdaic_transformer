# IP Synthetic Experiment

## Dataset
- Train samples: `2048`
- Val samples: `512`
- Window size: `32`
- Prefix length: `/24`
- Train positive rate: `0.2998`
- Val positive rate: `0.2988`

## Results

| Model | AUROC | F1 | Precision | Recall | p-adic corr | hierarchy gap | gate | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.5000 | 0.0000 | - | - | - | - | - | 0.02 |
| isolation_forest | 0.5201 | 0.4602 | 0.2988 | 1.0000 | - | - | - | 0.14 |
| vanilla_transformer | 0.5052 | 0.2283 | 0.2871 | 0.1895 | - | - | - | 0.95 |
| padic_attention_true | 0.5538 | 0.4657 | 0.3217 | 0.8431 | 0.0681 | 0.0127 | 0.5001 | 53.18 |
| padic_attention_shuffled | 0.5233 | 0.4444 | 0.3009 | 0.8497 | 0.2080 | 0.0158 | 0.5000 | 51.17 |
| padic_attention_random | 0.5078 | 0.3702 | 0.3206 | 0.4379 | 0.2284 | 0.0193 | 0.4999 | 48.58 |

## Notes

- `padic_attention_true` keeps MSB-first IP prefix bits.
- `padic_attention_shuffled` randomly permutes the unique IP-token vocabulary.
- `padic_attention_random` remaps each unique IP token to random 32-bit digits.

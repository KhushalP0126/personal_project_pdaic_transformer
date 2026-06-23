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
| vanilla_transformer | 0.4680 | 0.2716 | 0.2573 | 0.2876 | - | - | - | 0.54 |
| padic_attention_true | 0.5750 | 0.4715 | 0.3139 | 0.9477 | 0.0881 | 0.0239 | 0.4999 | 9.44 |
| padic_attention_shuffled | 0.4656 | 0.2283 | 0.2871 | 0.1895 | 0.1947 | 0.0326 | 0.5001 | 10.32 |
| padic_attention_random | 0.4657 | 0.4602 | 0.2988 | 1.0000 | 0.2114 | 0.0321 | 0.4999 | 11.22 |

## Notes

- `padic_attention_true` keeps MSB-first IP prefix bits.
- `padic_attention_shuffled` randomly permutes the unique IP-token vocabulary.
- `padic_attention_random` remaps each unique IP token to random 32-bit digits.

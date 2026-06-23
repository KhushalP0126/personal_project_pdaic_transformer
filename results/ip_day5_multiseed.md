# IP Day 5 Multi-Seed Validation

## Per-seed AUROC

| Seed | Logistic | IsolationForest | Vanilla | True 2-adic | Shuffled | Random | True - Vanilla | True - Best Control |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260504 | 0.5570 | 0.5452 | 0.5339 | 0.6559 | 0.5111 | 0.5353 | 0.1219 | 0.1205 |
| 20260505 | 0.5274 | 0.5595 | 0.5799 | 0.5404 | 0.4707 | 0.5116 | -0.0394 | 0.0288 |
| 20260506 | 0.5000 | 0.4672 | 0.5930 | 0.6057 | 0.4883 | 0.5465 | 0.0127 | 0.0592 |

## Mean ± std

| Model | AUROC | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.5282 ± 0.0285 | 0.2674 ± 0.2318 | nan ± nan | nan ± nan | nan ± nan |
| isolation_forest | 0.5240 ± 0.0497 | 0.4602 ± 0.0000 | 0.2988 ± 0.0000 | 1.0000 ± 0.0000 | 0.2988 ± 0.0000 |
| vanilla_transformer | 0.5689 ± 0.0310 | 0.4601 ± 0.0163 | 0.3087 ± 0.0040 | 0.9129 ± 0.1290 | 0.3626 ± 0.0539 |
| padic_attention_true | 0.6007 ± 0.0579 | 0.4732 ± 0.0243 | 0.3263 ± 0.0173 | 0.8649 ± 0.0887 | 0.4251 ± 0.0472 |
| padic_attention_shuffled | 0.4900 ± 0.0202 | 0.2367 ± 0.2162 | 0.1893 ± 0.1640 | 0.3638 ± 0.4032 | 0.5384 ± 0.1760 |
| padic_attention_random | 0.5312 ± 0.0178 | 0.3875 ± 0.1231 | 0.3174 ± 0.0281 | 0.7015 ± 0.4444 | 0.4336 ± 0.1895 |

## Paper-critical gaps

| Comparison | Mean ± std |
|---|---:|
| True 2-adic - vanilla | 0.0317 ± 0.0823 |
| True 2-adic - shuffled | 0.1106 ± 0.0380 |
| True 2-adic - random | 0.0695 ± 0.0467 |
| True 2-adic - best control | 0.0695 ± 0.0467 |

## Win counts

- True 2-adic beats vanilla in `2/3` seeds.
- True 2-adic beats best hierarchy control in `3/3` seeds.

## Attention diagnostics

| Metric | Mean ± std |
|---|---:|
| p-adic gate | 0.4992 ± 0.0006 |
| p-adic attention corr | 0.0798 ± 0.0118 |
| hierarchy gap | 0.0215 ± 0.0024 |

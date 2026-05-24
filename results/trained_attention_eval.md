# Trained Attention Evaluation

- Checkpoint: `results/checkpoints/best.pt`
- Dataset: `hierarchy_rules`
- Device: `cpu`

| Variant | AUROC | F1 | hierarchy corr | hierarchy gap | p-adic gate |
|---|---:|---:|---:|---:|---:|
| true | 0.5211 | 0.4081 | 0.2986 | 0.0079 | 0.1208 |
| shuffled | 0.4732 | 0.4458 | 0.1858 | 0.0040 | 0.1208 |
| random | 0.4858 | 0.4462 | 0.0807 | 0.0021 | 0.1208 |

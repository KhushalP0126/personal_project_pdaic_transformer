# IP Day 4 Tuning

| Config | True AUROC | Vanilla | Shuffled | Random | True - Vanilla | True - Best Control | Gate | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_w16_d64_l1_drop01_e1 | 0.5750 | 0.4680 | 0.4656 | 0.4657 | 0.1070 | 0.1093 | 0.4999 | 9.44 |
| epochs3_w16_d64_l1_drop01 | 0.6559 | 0.5339 | 0.5111 | 0.5352 | 0.1220 | 0.1207 | 0.4986 | 31.67 |
| d128_w16_l1_drop01_e1 | 0.5301 | 0.5271 | 0.5432 | 0.5145 | 0.0030 | -0.0131 | 0.4997 | 15.35 |
| layers2_w16_d64_drop01_e1 | 0.5723 | 0.4816 | 0.5062 | 0.5398 | 0.0907 | 0.0325 | 0.4996 | 21.99 |
| drop02_w16_d64_l1_e1 | 0.5723 | 0.4620 | 0.4671 | 0.4639 | 0.1103 | 0.1051 | 0.5000 | 10.15 |
| window32_d64_l1_drop01_e1 | 0.5538 | 0.5052 | 0.5233 | 0.5078 | 0.0486 | 0.0305 | 0.5001 | 53.18 |

## Selection

- Best true PDAIC AUROC: `epochs3_w16_d64_l1_drop01` at `0.6559`.
- Best hierarchy-control gap: `epochs3_w16_d64_l1_drop01` at `0.1207`.

Use the AUROC winner only if it still beats shuffled/random. Otherwise prefer the clean-control-gap winner for Day 5.

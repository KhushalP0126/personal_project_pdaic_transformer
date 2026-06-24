# Final Summary

## Variant meanings

| Variant | Hensel embedding | Explicit p-adic attention bias | Purpose |
|---|---|---|---|
| `standard_transformer` | no | no | raw-token baseline with OOV fallback |
| `flat_digit_transformer` | no | no | flat digit/bit projection baseline |
| `hensel_only` | yes | no | tests whether Hensel coordinates alone help |
| `hensel_padic_sigmoid` | yes | yes, old sigmoid gate | tests positive-only ultrametric bias |
| `hensel_padic_signed_alpha` | yes | yes, signed alpha | tests whether the model should attract, ignore, or oppose p-adic closeness |

## AUROC mean +- std

| Task | Best model | Standard | Flat digit | Hensel-only | Old gate | Signed alpha | Takeaway |
|---|---|---:|---:|---:|---:|---:|---|
| Simple synthetic | hensel_padic_signed_alpha | 0.5000 +- 0.0000 | 0.5372 +- 0.0209 | 0.5703 +- 0.0634 | 0.5811 +- 0.0461 | 0.6065 +- 0.0191 | signed alpha wins while keeping explicit bias near zero |
| Transition | standard_transformer | 0.5000 +- 0.0000 | 0.4802 +- 0.0136 | 0.4877 +- 0.0299 | 0.4901 +- 0.0219 | 0.4982 +- 0.0075 | transition task weakens the gain; signed alpha only edges the old gate |
| Cross-generator simple->transition | hensel_padic_signed_alpha | 0.5000 +- 0.0000 | 0.4911 +- 0.0287 | 0.4876 +- 0.0186 | 0.5003 +- 0.0402 | 0.5016 +- 0.0257 | generator shift remains the main failure mode |
| Cross-generator transition->simple | flat_digit_transformer | 0.5000 +- 0.0000 | 0.5148 +- 0.0467 | 0.4744 +- 0.0211 | 0.4823 +- 0.0164 | 0.5065 +- 0.0389 | reverse generator shift wipes out the structured advantage |
| Realistic proxy | standard_transformer | 0.7446 +- 0.0530 | 0.7315 +- 0.0741 | 0.7186 +- 0.0657 | 0.6926 +- 0.0488 | 0.7100 +- 0.0582 | realistic proxy favors a weaker explicit bias than the old gate |

## Signed-alpha diagnostics

| Task | alpha | raw alpha | alpha grad | p-adic corr | hierarchy gap | content std | p-adic std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Simple synthetic | -0.0032 +- 0.0004 | -0.0032 +- 0.0004 | 0.0032 +- 0.0008 | -0.0783 +- 0.0135 | -0.0182 +- 0.0016 | 0.1351 +- 0.0177 | 0.0120 +- 0.0009 |
| Transition | 0.0009 +- 0.0008 | 0.0009 +- 0.0008 | 0.0004 +- 0.0001 | 0.0081 +- 0.0179 | -0.0021 +- 0.0019 | 0.2121 +- 0.0149 | 0.0452 +- 0.0026 |
| Cross-generator simple->transition | -0.0044 +- 0.0002 | -0.0044 +- 0.0002 | 0.0035 +- 0.0005 | -0.0238 +- 0.0161 | -0.0026 +- 0.0069 | 0.2409 +- 0.0180 | 0.0459 +- 0.0057 |
| Cross-generator transition->simple | 0.0002 +- 0.0007 | 0.0002 +- 0.0007 | 0.0005 +- 0.0001 | -0.0607 +- 0.0120 | -0.0150 +- 0.0050 | 0.1256 +- 0.0096 | 0.0106 +- 0.0006 |
| Realistic proxy | -0.0010 +- 0.0001 | -0.0010 +- 0.0001 | 0.0084 +- 0.0027 | -0.1774 +- 0.0226 | -0.0216 +- 0.0074 | 0.2175 +- 0.0211 | 0.1969 +- 0.0134 |

## Comparison gaps

| Task | Hensel-standard | Hensel-flat | Old gate-Hensel | Signed-Hensel | Signed-old gate | Signed-flat |
|---|---:|---:|---:|---:|---:|---:|
| Simple synthetic | 0.0703 +- 0.0634 | 0.0331 +- 0.0752 | 0.0108 +- 0.0266 | 0.0362 +- 0.0694 | 0.0254 +- 0.0585 | 0.0693 +- 0.0060 |
| Transition | -0.0123 +- 0.0299 | 0.0075 +- 0.0238 | 0.0024 +- 0.0080 | 0.0105 +- 0.0251 | 0.0081 +- 0.0174 | 0.0180 +- 0.0062 |
| Cross-generator simple->transition | -0.0124 +- 0.0186 | -0.0035 +- 0.0143 | 0.0127 +- 0.0401 | 0.0140 +- 0.0076 | 0.0013 +- 0.0395 | 0.0105 +- 0.0093 |
| Cross-generator transition->simple | -0.0256 +- 0.0211 | -0.0404 +- 0.0494 | 0.0079 +- 0.0300 | 0.0321 +- 0.0479 | 0.0241 +- 0.0226 | -0.0083 +- 0.0854 |
| Realistic proxy | -0.0260 +- 0.0418 | -0.0129 +- 0.0124 | -0.0260 +- 0.0351 | -0.0086 +- 0.0112 | 0.0174 +- 0.0240 | -0.0215 +- 0.0159 |

## Baseline note

The raw-token `standard_transformer` builds its token vocabulary from the training split and maps unseen eval ids to one OOV token.
This deliberately tests raw-token generalization without digit sharing.
It is a defensible inductive-bias baseline, but not a claim that standard Transformers fail in general.

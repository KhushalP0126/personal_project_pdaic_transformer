# Final Summary

| Task | Best model | Standard | Hensel-only | Old gate | Signed alpha | Takeaway |
|---|---|---:|---:|---:|---:|---|
| Simple synthetic | hensel_padic_signed_alpha | 0.5000 | 0.5846 | 0.6089 | 0.6260 | signed alpha wins while collapsing the explicit bias near zero |
| Transition | hensel_padic_signed_alpha | 0.5000 | 0.4622 | 0.5205 | 0.5252 | signed alpha slightly beats the old gate while keeping alpha near zero |
| Cross-generator simple->transition | hensel_only | 0.5000 | 0.5400 | 0.4531 | 0.4522 | transfer is weak under generator shift |
| Cross-generator transition->simple | standard_transformer | 0.5000 | 0.4935 | 0.4330 | 0.4503 | transfer is weak under reverse generator shift |
| Realistic proxy | standard_transformer | 0.7485 | 0.6895 | 0.6013 | 0.7484 | signed alpha matches the standard transformer by neutralizing explicit bias |

## Signed-alpha diagnostics

| Task | alpha | raw alpha | alpha grad | p-adic corr | hierarchy gap | content std | p-adic std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Simple synthetic | -0.0029 | -0.0029 | 0.0025 | -0.0533 | -0.0081 | 0.1428 | 0.0105 |
| Transition | -0.0007 | -0.0007 | 0.0004 | 0.0025 | -0.0035 | 0.2012 | 0.0431 |
| Cross simple->transition | -0.0034 | -0.0034 | 0.0021 | -0.0374 | -0.0061 | 0.2327 | 0.0444 |
| Cross transition->simple | 0.0013 | 0.0013 | 0.0006 | -0.0738 | -0.0162 | 0.1232 | 0.0099 |
| Realistic proxy | 0.0000 | 0.0000 | 0.0059 | -0.1607 | -0.0197 | 0.1620 | 0.1942 |

# Limitations

## No real routing corpus

The project still does not evaluate on a real BGP or production IP-traffic anomaly dataset. The current "realistic" row is an idle-heavy synthetic proxy, not a real routing benchmark.

## Generator dependence

The cross-generator rows are weak:

- simple->transition: best AUROC is only `0.5400`
- transition->simple: no structured variant beats `0.5000`

That means the hierarchy signal is not yet generator-invariant.

## Explicit bias is not the stable win

The old sigmoid gate underperforms on the realistic proxy (`0.6013`) and is not the best model in any transfer setting. Signed alpha helps mainly because it can shrink the explicit p-adic bias toward zero.

## Signed alpha did not go strongly negative

The most optimistic hypothesis was that the realistic proxy would drive alpha negative under hierarchy mismatch. It did not. Instead, alpha converged to approximately zero while the hierarchy diagnostics stayed negative. That is still informative, but weaker than a true learned reversal of the bias.

## CPU-only budget

All reported study numbers come from the CPU configuration:

- 3 epochs
- `d_model=64`
- `n_layers=1`
- modest train/validation sizes

This is good for controlled comparison, but it is not an exhaustive capacity study.

## Standard transformer baseline is intentionally raw-token

On the IP tasks, the standard transformer consumes token ids without Hensel structure. Its flat `0.5000` performance on the simple and transition synthetic tasks means the benchmark is heavily testing structure sharing across unseen addresses. That is a valid design choice for this paper, but it should be stated clearly so reviewers do not read it as a universal transformer failure claim.

## Claim boundary

The safe claim is:

> Hensel coordinates are useful on hierarchy-aligned IP tasks, while explicit ultrametric attention bias is conditional and often best when it can collapse toward zero.

The unsafe claims are:

- p-adic attention beats transformers in general
- the explicit attention kernel is the main source of gain
- the method generalizes to real traffic

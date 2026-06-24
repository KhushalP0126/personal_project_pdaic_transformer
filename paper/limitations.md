# Limitations

## No real routing corpus

The project still does not evaluate on a real BGP or production IP-traffic anomaly dataset. The current "realistic" row is an idle-heavy synthetic proxy, not a real routing benchmark.

## Generator dependence

The cross-generator rows are weak:

- simple->transition: signed alpha is only `0.5016 +- 0.0257`
- transition->simple: the best row is the flat digit baseline at `0.5148 +- 0.0467`

That means the hierarchy signal is not yet generator-invariant.

## Flat digit baseline is strong

The new flat digit baseline narrows the interpretation:

- simple synthetic: `0.5372 +- 0.0209`
- realistic proxy: `0.7315 +- 0.0741`

So the paper can no longer treat the gain as uniquely Hensel-derived. It has to separate general digit-structure benefit from specifically Hensel benefit.

## Explicit bias is not the stable win

The old sigmoid gate underperforms on average on the realistic proxy (`0.6926 +- 0.0488`) and is not the best model in any transfer setting. Signed alpha helps mainly because it can shrink the explicit p-adic bias toward zero.

## Signed alpha did not go strongly negative

The most optimistic hypothesis was that the realistic proxy would drive alpha negative under hierarchy mismatch. It did not. Instead, alpha converged to approximately zero (`-0.0010 +- 0.0001`) while the hierarchy diagnostics stayed negative. That is still informative, but weaker than a true learned reversal of the bias.

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

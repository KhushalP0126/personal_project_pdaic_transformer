# Personal Project: p-adic Transformer

This repo builds a p-adic anomaly detection pipeline around Hensel-coded token sequences, a transformer encoder, and a binary anomaly head. The current attention model is hybrid: learned content attention plus a gated p-adic hierarchy bias.

## Quick Explainer

If you are new to p-adic numbers and want a visual introduction before reading the rest of the project, start here:

- [YouTube explainer](https://www.youtube.com/watch?v=3gyHKCDq1YA)
- [YouTube explainer 2](https://www.youtube.com/watch?v=v9QTK7zBAhw)
- [In-depth p-adic and adic numbers playlist](https://www.youtube.com/playlist?list=PL8I7rVYxS9skcwABs4kBDkJnYExQxZsE_)

## Hypothesis

The central claim is that anomaly detection improves when the model can use both:

- learned sequence/content rules
- p-adic hierarchy structure from Hensel-coded tokens

The repo is designed to test whether the hierarchy signal is real, whether it survives against shuffled or random hierarchy controls, and whether it still matters when the model is forced to learn a useful ranking signal first.

## Current Status

The codebase is in a much better experimental state than the initial synthetic runs, but it is still not publication-ready.

What is already in place:

- hybrid attention with `content_logits + gated_padic_logits`
- no-op retry logic for synthetic and realistic attack injection
- exact rank-based AUROC
- validation score-gap diagnostics
- hierarchy metrics in attention evaluation
- hierarchy-rule training dataset
- baseline suite with true, shuffled, and random hierarchy controls

What remains open:

- stronger evidence that p-adic structure is the cause of improvement
- multiple seeds with mean and standard deviation
- clean randomized-hierarchy ablations
- stronger real-dataset evidence
- a tighter paper narrative

## What This Repo Is Testing

### 1. The data hypothesis

Normal windows should follow a stable p-adic hierarchy. Anomalies should break that hierarchy by injecting tokens from a different class or subtree.

### 2. The model hypothesis

The model should not rely on p-adic structure alone. It should combine learned content attention with p-adic bias so it can express both hierarchy and sequence rules.

### 3. The evaluation hypothesis

If p-adic structure matters, then:

- AUROC should improve over non-p-adic baselines
- hierarchy metrics should move in the expected direction
- shuffled/random hierarchy should reduce or erase the gain

## Data

### Synthetic windows

The default synthetic dataset creates class clusters where tokens share low-order Hensel prefixes. A normal window is a contiguous slice from the generated stream. An attack replaces a segment with tokens from another class.

### Hierarchy-rule dataset

`--hierarchy-rule-dataset` turns the benchmark into a direct hierarchy test:

- normal windows stay inside the same p-adic subtree for several steps
- anomalous windows jump to a different subtree

This is the cleanest dataset for testing whether the hierarchy itself helps.

### Realistic dataset path

`--realistic-dataset` keeps the synthetic generator but makes it closer to hardware traces:

- idle-heavy windows
- low attack rates
- frequency-weighted loss
- retry logic for no-op mutations

### Open datasets

`scripts/run_open_dataset.py` can benchmark ADFA-LD or BETH. It reports:

- real attack rate
- estimated `pos_weight`
- ultrametric verdict
- IsolationForest comparison
- repeated runs across seeds

## Model

### Hensel embedding

Each digit position has its own embedding table. The model sums embeddings across the `r` digit positions.

### Hybrid attention

The current attention path combines:

- learned query-key content logits
- a gated p-adic similarity term

That design keeps hierarchy available without forcing the entire model to depend on it.

### Anomaly head

The head predicts one binary logit per window: normal vs anomalous.

### Training objective

BCE trains the anomaly label. Contrastive loss encourages examples with the same label to be closer and different labels to be farther apart. The recommended workflow is still to start with `--alpha 0.0` and confirm the detector can rank windows first.

## Metrics

### Primary metrics

- AUROC is the first sanity check
- best F1 is useful after threshold search
- recall and false-positive rate matter most for deployment

The code now uses exact rank-based AUROC rather than an approximate threshold sweep.

### Training diagnostics

Validation reports:

- normal and anomaly score means
- score gap
- best threshold
- precision, recall, F1, and false-positive rate

### Attention diagnostics

The attention path reports:

- p-adic attention correlation
- same-cluster attention
- different-cluster attention
- hierarchy gap
- depth-specific gaps for shared p-adic prefixes
- padic gate value

## Baselines

The baseline runner includes:

- majority-class predictor
- logistic regression
- MLP
- vanilla transformer
- Hensel transformer
- p-adic attention with true hierarchy
- p-adic attention with shuffled hierarchy
- p-adic attention with random hierarchy

Those controls are the main check against the claim that the gain comes from p-adic structure rather than just extra parameters.

## Open Questions

The repo still needs stronger answers to these questions:

- Does p-adic encoding help beyond standard sequence modeling?
- Does the gain survive hierarchy shuffling?
- Does the gain survive random hierarchy labels?
- Does it transfer to real datasets?
- Are the reported gains stable across seeds?

## Recommended Experiment Order

1. Start with BCE-first hybrid attention.
2. Run the hierarchy-rule benchmark.
3. Run the baseline suite.
4. Evaluate a trained checkpoint against true, shuffled, and random hierarchy.
5. Move to the realistic dataset path.
6. Validate on ADFA-LD or BETH.
7. Repeat the main runs across multiple seeds.

## Quick Start

```bash
make setup
make smoke
```

Recommended GPU path:

```bash
make train
```

Hierarchy-rule benchmark and controls:

```bash
make hierarchy
make baselines
make eval
```

Untrained hierarchy sweep:

```bash
make sweep
```

Realistic training:

```bash
make realistic
```

Real dataset benchmark:

```bash
make adfa
```

## Validation

Run the standard-library tests:

```bash
make test
```

## Make Targets

- `make cpu` and `make gpu` run the benchmark path, not training.
- `make smoke` is the local CPU sanity-training path.
- `make train` is the recommended first GPU run for the hybrid attention model.
- `make vanilla` runs the standard transformer GPU training path without PDAIC attention.
- `make hierarchy` runs the hierarchy-rule benchmark training path.
- `make realistic` runs the realistic idle-heavy training path.
- `make baselines` runs the majority, linear, MLP, transformer, and hierarchy-control baselines.
- `make eval` evaluates a trained checkpoint against true, shuffled, and random hierarchy controls.
- `make sweep` reports sparsity plus hierarchy-alignment metrics across `p` on an untrained model.
- `make adfa` downloads and benchmarks ADFA-LD.
- `make beth` downloads and benchmarks BETH.
- `make adfa-stats` prints ADFA-LD stats without training.

## References

- [Learning with the p-adics](https://arxiv.org/abs/2512.22692) (December 2025)
  - `Identifier:` arXiv:2512.22692
  - `Relevance:` This paper establishes the mathematical foundation for applying the non-Archimedean space of $p$-adic numbers ($\mathbb{Q}_p$) to machine learning. It highlights the inherent hierarchical structure of p-adics, which justifies their use for representation learning and anomaly detection in this architecture.
- [pASCNN: p-adic Sheaf-Coherence Neural Network](https://github.com/kaifczxc-lab/pASCNN) (April 2026)
  - `Identifier:` GitHub Repository (kaifczxc-lab/pASCNN)
  - `Relevance:` Explores the implementation of p-adic ultrametrics to structure logical inference in complex spaces. It provides a highly relevant conceptual baseline for substituting standard attention mechanisms with p-adic structured logic.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (June 2017)
  - `Authors:` Vaswani, A., et al.
  - `Identifier:` arXiv:1706.03762
  - `Relevance:` The foundational paper for the baseline Transformer architecture. It serves as the standard comparative baseline against the custom p-adic modifications introduced in this repository.

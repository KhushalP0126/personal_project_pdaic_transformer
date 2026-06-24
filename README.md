# Personal Project: p-adic Transformer

This repo builds a p-adic anomaly detection pipeline around Hensel-coded token sequences, a transformer encoder, and a binary anomaly head. The current paper-facing direction is a controlled study that separates Hensel embedding from explicit p-adic attention bias.

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

The current direction is an IP-prefix anomaly-detection paper:

```text
Paper: 2-adic Prefix-Aware Transformer for IP Traffic Anomaly Detection
Goal: workshop/student/arXiv-style draft
Compute: CPU only
Primary dataset path: synthetic IPv4 prefix windows
Primary command: make ip-study-cpu
```

The project is now using the older Hensel/syscall-style synthetic work as supporting infrastructure, not as the main paper target. The active claim is modest:

```text
2-adic structure can help on hierarchy-aligned IP tasks, but the gain depends on whether it enters as coordinates, as explicit attention bias, or both.
```

The codebase is in a usable experimental state for the IP-prefix direction, but it is still not publication-ready.

## Attention Mechanism

There are two separate p-adic mechanisms in the current study:

| Mechanism | Code variant | Meaning |
|---|---|---|
| Hensel digit embedding | `hensel_only` | p-adic coordinate representation |
| Ultrametric attention bias | `hensel_padic_sigmoid`, `hensel_padic_signed_alpha` | p-adic relational prior between tokens |

The attention path itself is:

```text
content_logits = QK^T / sqrt(d)
padic_logits = normalized p-adic valuation matrix
attention_logits = content_logits + alpha * padic_logits
```

The three explicit bias modes are:

| Mode | Alpha behavior | Interpretation |
|---|---|---|
| `none` | `alpha = 0` | Hensel-only, no explicit p-adic attention bias |
| `sigmoid` | `alpha in [0,1]` | old positive-only p-adic gate |
| `signed_alpha` | `alpha = alpha_max * tanh(raw_alpha)` | can attract, ignore, or oppose p-adic closeness |

The important result is that signed alpha mostly improves robustness by letting the explicit bias shrink back toward zero when it is not useful.

## Reproducibility

### Environment setup

```bash
make setup
```

### Sanity checks

Run the repo checks before comparing results:

```bash
make test
make check-study
```

If the repo is already set up and you want to avoid any reinstall step:

```bash
make test-unit
```

### Reproduce the current characterization study

```bash
make ip-study-cpu
```

This writes:

```text
results/final_summary.json
results/final_summary.md
```

The controlled study compares:

- `standard_transformer`
- `flat_digit_transformer`
- `hensel_only`
- `hensel_padic_sigmoid`
- `hensel_padic_signed_alpha`

across:

- simple synthetic IP-prefix anomalies
- harder transition-rule IP anomalies
- simple->transition transfer
- transition->simple transfer
- realistic idle-heavy proxy data

The raw-token `standard_transformer` is intentionally strict: it builds a token vocabulary from the training split and sends unseen validation addresses to one OOV token. That baseline tests raw-token generalization without digit sharing. It is part of the inductive-bias question, not a claim that standard Transformers fail in general.

### CPU study budget

All final study numbers in `results/final_summary.*` were run on CPU with:

- 3 seeds: `20260504`, `20260505`, `20260506`
- 3 epochs
- `d_model=64`
- `n_layers=1`
- `n_heads=4`
- `train_samples=2048`
- `val_samples=512`

### Claim boundary

Supported:

- structured digit or prefix features help on the simple aligned synthetic IP task
- signed alpha is safer than the old positive-only gate
- explicit p-adic attention bias is fragile
- transfer remains weak

Not supported:

- p-adic attention is the main mechanism
- p-adic Transformers beat standard Transformers generally
- the method generalizes to real traffic
- routing efficiency improvement

### Train and evaluate the Hensel hierarchy model

CPU sanity training:

```bash
make smoke
```

Recommended accelerator-backed training path:

```bash
make train
```

Evaluate a trained checkpoint against hierarchy controls:

```bash
make eval
```

What is already in place:

- IPv4-to-32-bit binary digit pipeline with MSB-first prefix semantics
- synthetic IP-prefix anomaly dataset with prefix jumps, spoofed prefixes, and route-leak style anomalies
- CPU IP experiment runner comparing vanilla, true 2-adic, shuffled 2-adic, random 2-adic, and simple baselines
- hybrid attention with `content_logits + gated_padic_logits`
- no-op retry logic for synthetic and realistic attack injection
- exact rank-based AUROC
- validation score-gap diagnostics
- hierarchy metrics in attention evaluation
- hierarchy-rule training dataset
- baseline suite with true, shuffled, and random hierarchy controls

## Current Result Read

The current repo result should be read from `results/final_summary.md` and
`results/final_summary_by_seed.md`.

The safe summary is:

- structured digit features help on the simple aligned synthetic IP task
- signed alpha is better than the old positive-only gate there
- the signed-alpha model does that while keeping `alpha` near zero
- transition and generator-shift rows are weak
- realistic-proxy results favor the non-p-adic baselines on average

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

`scripts/run_open_dataset.py` can benchmark BETH. It reports:

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

## Potential Uses of PDAIC Numbers

The active direction is IP-prefix anomaly detection. The reusable idea is broader: encode a discrete hierarchy as adic digits, then let attention combine learned content rules with hierarchy-aware bias.

| Application domain | Readiness | Codebase components reused | Required changes | Estimated pivot time |
|---|---:|---|---|---|
| IP routing and network analytics | 95% | Core transformer, p-adic attention, IP-prefix dataset, experiment runner, and classification head. | Add raw PCAP/network-log ingestion after the synthetic IP result is stable. | active target |
| Lexical semantics and WordNet | 75% | Core transformer, p-adic attention, and classification head. | Replace the syscall tokenizer with a parser that maps words to WordNet tree/path IDs. | short |
| Genomics and phylogenetics | 45% | Token embedding layer, classification head, and training loop. | Modify attention to accept a supplied evolutionary-tree distance matrix instead of computing shared-prefix distances from sequence digits. | medium |
| Document retrieval and Wasserstein-style matching | 15% | Token embedding layer and some raw attention math. | Replace the classification head, rewrite the objective, build a Siamese/twin-network setup, and add optimal-transport solvers. | 1-2 weeks |

IP routing is no longer just a pivot; it is the current paper path. IP addresses are hierarchical discrete strings, so `p=2, r=32` gives a natural prefix representation. Lexical semantics is the next most plausible future pivot because WordNet already gives a tree structure; the main work would be data preparation rather than model surgery.

## Quick Start

```bash
make setup
make test
make smoke
```

For the harder CPU transition check:

```bash
make ip-transition-cpu
```

For the main GPU training path:

```bash
make train
```

## Make Targets

### CPU First

- `make setup` creates `.venv` and installs the editable package.
- `make test` runs the unit test suite.
- `make smoke` runs the local CPU sanity-training path.
- `make cpu` runs the local CPU benchmark path, not training.
- `make cpu-all-1epoch` runs one CPU epoch across vanilla, PDAIC synthetic, hierarchy-rule, realistic, and baseline paths with progress output.
- `make ip-cpu` runs the CPU IP-prefix synthetic experiment.
- `make ip-transition-cpu` runs the harder transition-based CPU IP experiment.
- `make ip-study-cpu` runs the multi-seed characterization study and writes `final_summary`.
- `make test-unit` runs the unit tests without re-entering package setup.
- `make check-study` compile-checks the characterization study runner.
- `make clean` removes caches, checkpoints, and generated outputs.

Common CPU-facing scripts now alternate their default output slot between:

- `results/report.json` / `results/report.md`
- `results/report2.json` / `results/report2.md`

That only applies when you keep the script defaults. If you pass explicit output
paths, those are still honored.

### GPU Second

- `make gpu` runs the accelerator benchmark path, not training.
- `make train` runs the recommended BCE-first GPU attention training path.
- `make vanilla` runs the standard transformer GPU training path without PDAIC attention.
- `make hierarchy` runs the hierarchy-rule attention training path.
- `make realistic` runs the realistic idle-heavy attention training path.

### Analysis

- `make primes` runs vanilla `p=3,5,7` comparisons.
- `make pdaic-primes` runs PDAIC attention `p=3,5,7` comparisons.
- `make compare-analysis` compares the vanilla and PDAIC prime sweep logs.
- `make analysis` runs vanilla primes, PDAIC primes, and the comparison report.
- `make baselines` runs the majority, linear, MLP, transformer, and hierarchy-control baselines.
- `make eval` evaluates a trained checkpoint against true, shuffled, and random hierarchy controls.
- `make sweep` reports sparsity plus hierarchy-alignment metrics across `p` on an untrained model.
- `make threshold` runs the GPU training path with threshold tuning output.
- `make diagnose` runs the learning-vs-overfit diagnostic.
- `make audit` audits synthetic datasets for imbalance, leakage, train/validation overlap, and obvious artifacts.

### Open Datasets

- `make beth` downloads and benchmarks BETH. It needs Kaggle credentials or local BETH CSV files under `data/beth`.

### 2-Adic / Hardware

- `make int8` verifies unsigned INT8 arithmetic against truncated 2-adic arithmetic.
- `make hardware` is an alias for `make int8`.

## Validation

Run the standard-library tests:

```bash
make test
```

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

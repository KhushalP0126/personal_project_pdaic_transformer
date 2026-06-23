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

The current direction is an IP-prefix anomaly-detection paper:

```text
Paper: 2-adic Prefix-Aware Transformer for IP Traffic Anomaly Detection
Goal: workshop/student/arXiv-style draft
Compute: CPU only
Primary dataset path: synthetic IPv4 prefix windows
Primary command: make ip-cpu
```

The project is now using the older Hensel/syscall-style synthetic work as supporting infrastructure, not as the main paper target. The active claim is modest:

```text
2-adic attention is a useful inductive bias for IP-prefix anomaly detection.
```

The codebase is in a usable experimental state for the IP-prefix direction, but it is still not publication-ready.

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

What remains open:

- Day 4 tuning on the IP-prefix task
- Day 5 multi-seed table for the best small CPU config
- stronger evidence that p-adic structure is the cause of improvement
- multiple seeds with mean and standard deviation
- clean randomized-hierarchy ablations
- eventual validation on BETH or a real IP/network-traffic dataset
- a tighter paper narrative

## Current Evidence From Results

The current results do **not** confirm the central hypothesis yet. They show useful signal, but the evidence is still weak and mixed.

### IP-prefix Day 3 result

`results/ip_day3_first.json` is the first useful result for the IP-prefix paper direction. The experiment uses `p=2`, `r=32`, `/24` prefixes, 2048 train windows, 512 validation windows, one seed, and one CPU epoch.

| Model | AUROC | F1 |
|---|---:|---:|
| logistic_regression | 0.5503 | 0.4140 |
| isolation_forest | 0.5452 | 0.4602 |
| vanilla_transformer | 0.4680 | 0.2716 |
| padic_attention_true | 0.5750 | 0.4715 |
| padic_attention_shuffled | 0.4656 | 0.2283 |
| padic_attention_random | 0.4657 | 0.4602 |

This is a green light to continue the IP-prefix experiment. The important ordering is:

```text
true 2-adic > vanilla
true 2-adic > shuffled
true 2-adic > random
```

That suggests the real IP-prefix hierarchy is contributing useful signal. When the hierarchy is destroyed by shuffled or random remaps, AUROC falls back near the vanilla transformer.

The result is not final evidence. The margin over logistic regression and IsolationForest is small, and the run is only one seed and one epoch. The PDAIC gate also stayed near initialization (`padic_gate ~= 0.5`), so Day 4 should test whether more epochs or slightly larger models make the p-adic branch move.

### Central finding

`results/trained_attention_eval.json` is the most important hierarchy-control result so far:

| Variant | AUROC |
|---|---:|
| true | 0.521 |
| shuffled | 0.473 |
| random | 0.486 |

The true hierarchy checkpoint is better than shuffled and random hierarchy, which is the expected direction. The gap is small, and the absolute AUROC values are weak. This checkpoint came from the small CPU smoke run (`d_model=64`), so it is not conclusive. It is a signal to re-test with a stronger checkpoint, not proof.

### Baseline report

`results/baseline_report.json` is the most concerning result:

| Model | AUROC |
|---|---:|
| standard_transformer | 0.611 |
| isolation_forest | 0.584 |
| padic_attention_true | 0.500 |
| padic_attention_shuffled | 0.533 |
| padic_attention_random | 0.504 |

The p-adic attention model with the true hierarchy loses to the standard transformer and IsolationForest on the hierarchy-rule dataset. Shuffled hierarchy scoring higher than true hierarchy is the opposite of what the hypothesis requires. This comparison is not final because the baseline run used small configs and only 5 epochs, but it is the strongest warning sign in the current evidence.

### Prime comparisons

The vanilla prime sweep shows `p=3` performs best:

| Prime | AUROC |
|---:|---:|
| 3 | 0.688 |
| 5 | 0.668 |
| 7 | 0.648 |

The PDAIC attention sweep shows the same pattern with a stronger `p=3` result:

| Prime | AUROC |
|---:|---:|
| 3 | 0.730 |
| 5 | 0.652 |
| 7 | 0.656 |

This is the cleanest positive result so far. Smaller primes give deeper trees for fixed `r`, so `p=3` having the strongest hierarchy signal is plausible. PDAIC attention improves over the non-attention model at `p=3` by about four AUROC points in these runs.

### Realistic run

`results/training_log.json` reports a high AUROC around 0.84, but the run is not structurally strong evidence yet. The validation set has only 20 anomalies because the realistic attack fraction is `0.005`. Best F1 is low, and the model mostly predicts normal while occasionally catching a true anomaly.

The score gap does grow during training, which suggests real signal, but the AUROC estimate is noisy with so few positives. This run needs more validation anomalies or multiple seeds before it can support a strong claim.

### Threshold tuning

`results/tune_threshold.json` is currently the cleanest training run. It reaches about 0.743 AUROC on a more balanced synthetic setup. The ranking signal is real, but the train/validation loss gap is still large, which points to overfitting.

### Over/underfit diagnostic

`results/over_underfit.json` confirms the small-model problem. AUROC plateaus near 0.58 while validation loss diverges after the early epochs. The smaller `d_model=128` setup is underpowered or overfitting the available training distribution.

### Attention sweep

`results/p_base_attention_sweep.json` shows the untrained `padic_gate` near `0.1192`, which is `sigmoid(-2.0)`. Because that sweep is untrained, it mostly measures the prior. It also shows the current gate initialization is very conservative, so the p-adic term contributes only a small additive bias at startup.

### Critical problems

1. The p-adic gate may be too closed. Starting at `sigmoid(-2.0) ~= 0.119` makes the p-adic bias weak. A stronger test should initialize `padic_gate` at `0.0`, or add a regularizer that prevents the gate from staying near zero.
2. The baseline comparison is not yet fair. `make baselines` used small configs and 5 epochs, while the stronger training runs use more compute. Re-run baselines with matched epochs and model sizes before drawing conclusions.
3. Validation loss diverges in meaningful runs. Train loss drops while validation loss climbs, so the model is memorizing the synthetic distribution. This needs more diverse training data, stronger dropout, weight-decay tuning, or better hierarchy-rule generation.

### Next experiment order

1. Change `padic_gate` initialization from `-2.0` to `0.0`.
2. Re-run `make train`.
3. Re-run `make eval` against the new checkpoint.
4. Re-run `make baselines` with around 20 epochs or matched compute.
5. Treat `true >> shuffled/random` as the key evidence threshold. If that does not appear, the hierarchy hypothesis needs rethinking.

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
| Lexical semantics and WordNet | 75% | Core transformer, p-adic attention, and classification head. | Replace the syscall tokenizer with a parser that maps words to WordNet tree/path IDs. | 1 day |
| Genomics and phylogenetics | 45% | Token embedding layer, classification head, and training loop. | Modify attention to accept a supplied evolutionary-tree distance matrix instead of computing shared-prefix distances from sequence digits. | 3-5 days |
| Document retrieval and Wasserstein-style matching | 15% | Token embedding layer and some raw attention math. | Replace the classification head, rewrite the objective, build a Siamese/twin-network setup, and add optimal-transport solvers. | 1-2 weeks |

IP routing is no longer just a pivot; it is the current paper path. IP addresses are hierarchical discrete strings, so `p=2, r=32` gives a natural prefix representation. Lexical semantics is the next most plausible future pivot because WordNet already gives a tree structure; the main work would be data preparation rather than model surgery.

## Recommended Experiment Order

1. Use `make ip-cpu` as the reproducibility command for the IP-prefix experiment.
2. Day 4: tune only the small CPU grid: window size `16/32`, `d_model` `64/128`, layers `1/2`, dropout `0.1/0.2`, epochs `1/3`.
3. Keep `p=2, r=32`; do not spend time on `p=3,5,7` for the IP paper.
4. Compare true 2-adic attention against vanilla, shuffled 2-adic, random 2-adic, IsolationForest, and logistic regression.
5. Day 5: run three seeds on the best small CPU config and report AUROC/F1 mean and standard deviation.
6. Save attention diagnostics for every final run: `padic_attention_corr`, `hierarchy_gap`, `same_prefix_attention`, `diff_prefix_attention`, and `padic_gate`.
7. After the synthetic IP table is stable, validate on BETH or a real IP/network-traffic dataset.

## Quick Start

```bash
make setup
make test
make smoke
```

For the main GPU training path:

```bash
make train
```

## Make Targets

### Core

- `make setup` creates `.venv` and installs the editable package.
- `make test` runs the unit test suite.
- `make cpu` runs the local CPU benchmark path, not training.
- `make gpu` runs the CUDA benchmark path, not training.
- `make smoke` runs the local CPU sanity-training path.
- `make clean` removes caches, checkpoints, and generated outputs.

### Training

- `make train` runs the recommended BCE-first GPU attention training path.
- `make vanilla` runs the standard transformer GPU training path without PDAIC attention.
- `make hierarchy` runs the hierarchy-rule attention training path.
- `make realistic` runs the realistic idle-heavy attention training path.
- `make cpu-all-1epoch` runs one CPU epoch across vanilla, PDAIC synthetic, hierarchy-rule, realistic, and baseline paths with progress output.

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
- `make ip-cpu` runs the CPU IP-prefix synthetic experiment and writes `results/ip_synthetic.json` plus `results/ip_synthetic.md`.

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

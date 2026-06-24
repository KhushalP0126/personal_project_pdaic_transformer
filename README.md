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
Primary command: make ip-cpu
```

The project is now using the older Hensel/syscall-style synthetic work as supporting infrastructure, not as the main paper target. The active claim is modest:

```text
2-adic structure can help on hierarchy-aligned IP tasks, but the gain depends on whether it enters as coordinates, as explicit attention bias, or both.
```

The codebase is in a usable experimental state for the IP-prefix direction, but it is still not publication-ready.

## Reproducibility

### Environment setup

Use Python 3.10 or newer. The project is installable as an editable package, and
`requirements.txt` pins the runtime versions used for the checked-in result
files.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

The Makefile runs the same setup through:

```bash
make setup
```

### Sanity checks

Run the unit tests before comparing results:

```bash
make test
```

The fast CPU smoke path is:

```bash
make smoke
```

### Reproduce the current characterization study

The main CPU study command is:

```bash
make ip-study-cpu
```

Direct runner:

```bash
.venv/bin/python scripts/run_ip_characterization_study.py --device cpu
```

This writes:

```text
results/final_summary.json
results/final_summary.md
```

The controlled study compares:

- `standard_transformer`
- `hensel_only`
- `hensel_padic_sigmoid`
- `hensel_padic_signed_alpha`

across:

- simple synthetic IP-prefix anomalies
- harder transition-rule IP anomalies
- simple->transition transfer
- transition->simple transfer
- realistic idle-heavy proxy data

### Reproduce the earlier IP-prefix results

Day 3 first useful IP-prefix run:

```bash
.venv/bin/python scripts/run_ip_experiment.py \
  --device cpu \
  --train-samples 2048 \
  --val-samples 512 \
  --window-size 16 \
  --prefix-len 24 \
  --num-prefixes 32 \
  --attack-fraction 0.30 \
  --attack-min-len 1 \
  --attack-max-len 4 \
  --epochs 1 \
  --batch-size 128 \
  --lr 3e-4 \
  --d-model 64 \
  --n-heads 4 \
  --n-layers 1 \
  --d-digit 8 \
  --output-json results/ip_day3_first.json \
  --output-md results/ip_day3_first.md
```

Day 4 small CPU tuning sweep:

```bash
make ip-day4
```

The aggregate Day 4 report is written to:

```text
results/ip_day4_tuning.json
results/ip_day4_tuning.md
```

Day 5 multi-seed validation for the best Day 4 config:

```bash
make ip-day5
```

Fast first check:

```bash
make ip-day5-fast
```

The aggregate Day 5 report is written to:

```text
results/ip_day5_multiseed.json
results/ip_day5_multiseed.md
```

### Old gate ablation

The IP experiment runner now exposes fixed and learned gate variants directly:

```bash
# fixed gate ablations
.venv/bin/python scripts/run_ip_experiment.py --device cpu --fixed-padic-gate 0.0 ...
.venv/bin/python scripts/run_ip_experiment.py --device cpu --fixed-padic-gate 0.25 ...
.venv/bin/python scripts/run_ip_experiment.py --device cpu --fixed-padic-gate 0.5 ...
.venv/bin/python scripts/run_ip_experiment.py --device cpu --fixed-padic-gate 1.0 ...

# learned gate without the 0.5 pullback
.venv/bin/python scripts/run_ip_experiment.py --device cpu --gate-regularization-weight 0.0 ...
```

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

### Temperature initialization ablation

Soft p-adic valuation now uses flat temperature initialization by default. The
older prime-gap prior is opt-in so it must be ablated explicitly:

```bash
# Flat default
.venv/bin/python scripts/train_anomaly_detector.py \
  --attention \
  --device cpu \
  --p 3 \
  --r 8 \
  --d-model 64 \
  --n-heads 4 \
  --n-layers 2 \
  --ffn-dim 256 \
  --head-hidden 32 \
  --dropout 0.1 \
  --d-digit 8 \
  --window-size 16 \
  --attack-fraction 0.3 \
  --attack-min-len 2 \
  --attack-max-len 4 \
  --n-train 4096 \
  --n-val 512 \
  --samples 4096 \
  --classes 16 \
  --tokens-per-class 64 \
  --epochs 15 \
  --batch-size 64 \
  --lr 3e-4 \
  --warmup-epochs 2 \
  --num-workers 0 \
  --alpha 0.0 \
  --temperature-decay 0.0 \
  --save-every 999 \
  --checkpoint-dir results/temp_flat_checkpoints \
  --log-json results/temp_flat.json \
  --log-md results/temp_flat.md

# Prime-gap prior
.venv/bin/python scripts/train_anomaly_detector.py \
  --attention \
  --device cpu \
  --p 3 \
  --r 8 \
  --d-model 64 \
  --n-heads 4 \
  --n-layers 2 \
  --ffn-dim 256 \
  --head-hidden 32 \
  --dropout 0.1 \
  --d-digit 8 \
  --window-size 16 \
  --attack-fraction 0.3 \
  --attack-min-len 2 \
  --attack-max-len 4 \
  --n-train 4096 \
  --n-val 512 \
  --samples 4096 \
  --classes 16 \
  --tokens-per-class 64 \
  --epochs 15 \
  --batch-size 64 \
  --lr 3e-4 \
  --warmup-epochs 2 \
  --num-workers 0 \
  --alpha 0.0 \
  --temperature-decay 0.05 \
  --save-every 999 \
  --checkpoint-dir results/temp_prime_gap_checkpoints \
  --log-json results/temp_prime_gap.json \
  --log-md results/temp_prime_gap.md
```

Do not cite the prime-gap prior as evidence unless it beats the flat run under
the same seed and training budget.

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

- stronger evidence on real traffic or BETH-style data
- more seeds or confidence intervals for the vanilla comparison
- a cleaner explanation for why the learned gate stays near `0.5`
- a stronger causal story for when the p-adic branch helps versus hurts
- a tighter paper narrative

## Current Evidence From Results

The current synthetic IP-prefix results support a modest claim:

```text
true 2-adic attention consistently beats shuffled/random hierarchy controls,
but the margin over a vanilla transformer is still unstable.
```

That is enough for a workshop-style synthetic result, but it is not yet enough
for a strong general claim about p-adic attention.

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

The result is only a green light. It is one seed and one epoch, so it does not
carry the paper by itself.

### IP-prefix Day 4 tuning result

`results/ip_day4_tuning.json` picks the best small CPU configuration for Day 5:

| Config | True AUROC | Vanilla AUROC | Shuffled AUROC | Random AUROC | True - Best Control |
|---|---:|---:|---:|---:|---:|
| `epochs3_w16_d64_l1_drop01` | 0.6559 | 0.5339 | 0.5111 | 0.5352 | 0.1207 |

This is the strongest single-seed configuration found in the CPU tuning pass,
and it became the fixed Day 5 candidate.

### IP-prefix Day 5 multi-seed result

`results/ip_day5_multiseed.json` is the paper-critical synthetic result for the
current repo direction:

| Comparison | Result |
|---|---:|
| True 2-adic beats vanilla | `2/3` seeds |
| True 2-adic beats best shuffled/random control | `3/3` seeds |
| Mean true - vanilla AUROC gap | `+0.0317 ± 0.0823` |
| Mean true - best control AUROC gap | `+0.0695 ± 0.0467` |

The promising part is the control story: true 2-adic wins against shuffled or
random hierarchy in every seed. The weaker part is the vanilla comparison:
there is still one seed where vanilla wins, and the standard deviation on the
true-minus-vanilla gap is large.

### Gate ablation result

The gate is not broken. It is behaving like the code encourages: it starts at
`sigmoid(0.0) = 0.5`, and the default regularizer pulls it back toward `0.5`.
That means a gate near `0.5` is not evidence that the p-adic branch is unused.

The useful ablation is whether performance drops when the gate is fixed to
`0.0`. On the Day 5 setup, it does not. In fact, the strongest of the quick
ablations was:

| Variant | Mean true - vanilla | Mean true - best control |
|---|---:|---:|
| `fixed_gate_0` | `+0.0532` | `+0.0907` |
| `fixed_gate_025` | `+0.0407` | `+0.0783` |
| `fixed_gate_05` | `+0.0317` | `+0.0695` |
| `fixed_gate_1` | `+0.0255` | `+0.0638` |
| `learned_no_reg` | `+0.0317` | `+0.0695` |

The immediate conclusion is narrow but important: on this synthetic Day 5
budget, the hierarchy-control advantage is real, but the learned gate itself is
not the source of the gain. Forcing the gate to `0.0` performed best, while
removing the gate regularizer did not materially move the learned gate off its
`~0.5` baseline.

### Practical read on the evidence

1. The synthetic IP-prefix claim is strongest when phrased as a hierarchy
   control result: true 2-adic beats shuffled/random consistently.
2. The repo does not yet have a stable claim that true 2-adic reliably beats a
   vanilla transformer across seeds.
3. The gate ablation weakens any argument that the current learned gate is the
   mechanism behind the improvement.
4. The next serious step is real-data validation or a stronger causal ablation,
   not another broad synthetic sweep.

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
2. Use `make ip-day4` only when you need to re-run the small CPU tuning sweep.
3. Use `make ip-day5` for the three-seed synthetic validation table.
4. Use `--fixed-padic-gate` and `--gate-regularization-weight` for causal gate ablations before changing model internals again.
5. Save attention diagnostics for every final run: `padic_attention_corr`, `hierarchy_gap`, `same_prefix_attention`, `diff_prefix_attention`, and `padic_gate`.
6. Treat `true >> shuffled/random` as the primary synthetic success condition; treat `true > vanilla` as desirable but not yet stable.
7. After the synthetic IP table is stable, validate on BETH or a real IP/network-traffic dataset.

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
- `make ip-day4` runs the small CPU IP-prefix tuning pass.
- `make ip-day5` runs the three-seed CPU IP-prefix validation.
- `make ip-day5-fast` runs the smaller CPU Day 5 smoke pass.
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

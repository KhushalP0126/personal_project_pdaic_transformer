# Personal Project: p-adic Transformer

This repo builds a p-adic anomaly detection pipeline around Hensel-coded token sequences, a transformer encoder, and a binary anomaly head. The current attention model is a hybrid design: learned content attention plus a gated p-adic hierarchy bias.

## Big Picture

1. What problem am I solving?
   You are building an anomaly detector for system-event-like sequences. In the current code, it is not detecting real attacks yet; it detects synthetic "attack windows" inserted into Hensel-coded syscall-style sequences.

2. Why ultrametric/tree-like structure?
   Your generator creates token clusters where items in the same class share low-order Hensel prefixes. That means similarity is based on shared p-adic prefix depth, which naturally forms a tree-like hierarchy.

3. What does p-adic distance capture?
   It captures closeness by shared low-order digits, not ordinary geometric distance. Two tokens are close if they share a longer Hensel prefix.

4. Why transformer?
   Your model treats each window as a sequence and uses attention to learn patterns across tokens. In the current attention path, the model combines learned query-key logits with a gated p-adic similarity term, so it can use both learned sequence rules and hierarchy.

5. What counts as a rogue attack right now?
   A synthetic window where a segment of normal tokens is replaced by tokens from another class.

## Math

6. What is a p-adic number here?
   In your code, it is represented practically as fixed-width base-`p` digits.

7. What are Hensel digits?
   They are the digit tensor representation of tokens, shaped like `[items, r]` or `[batch, seq, r]`.

8. Why are low-order shared digits important?
   Your p-adic closeness depends on how many low-order digits match from index `0`.

9. What does ultrametric inequality mean?
   Distances obey a stronger triangle rule. Your code checks it using shared-prefix valuations: `v(x,z) >= min(v(x,y), v(y,z))`.

10. How does changing `p` affect representation?
    Changing `p` changes the digit alphabet size. `p=7` gives each digit 7 possible values, creating a richer branching structure than `p=3`.

11. What does `r=16` mean?
    Each token has 16 Hensel digits of precision.

12. Why might `p=7` perform differently?
    It gives more possible digit patterns per position, which may help separate classes, but can also make learning harder if data is sparse.

## Dataset

13. How is the synthetic dataset generated?
    It creates class centers, then makes tokens share a prefix with their class center and randomizes remaining digits.

14. What makes a normal sequence normal?
    A normal window is a contiguous slice from the generated token stream.

15. How is an attack injected?
    A random segment in the window is replaced with tokens from another class.

16. Is the attack realistic?
    Not yet. It is a useful synthetic proxy, but not a real syscall attack trace.

17. What assumptions does it make?
    It assumes normal behavior has stable p-adic cluster structure, and attacks look like class-disrupting substitutions.

18. What real dataset could replace it?
    Real syscall traces, process event logs, embedded device telemetry, or labeled intrusion datasets.

19. How do I run on a real dataset?
    Use `scripts/run_open_dataset.py` to download, parse, and benchmark ADFA-LD or BETH. The script reports the real attack rate, estimates `pos_weight`, checks the ultrametric assumption, compares IsolationForest against the p-adic model, and can repeat runs across multiple seeds.

## Model

19. How does Hensel embedding work?
    Each digit position has its own embedding table. The model sums embeddings across the `r` digit positions.

20. What does the transformer learn?
    It learns sequence patterns over embedded Hensel tokens. In the hybrid attention model, it also learns how strongly to use the p-adic hierarchy via a trainable gate instead of relying on hierarchy alone.

21. What does the anomaly head predict?
    One binary logit per window: normal vs anomalous.

22. Why BCE plus contrastive loss?
    BCE trains the anomaly label. Contrastive loss encourages p-adic structure to separate normal and attack windows. The recommended workflow is to start with `--alpha 0.0` and confirm BCE can learn ranking first, then reintroduce contrastive loss gradually.

23. What does contrastive loss encourage?
    Pairs with the same label should be closer; pairs with different labels should be farther apart.

24. Is the model deployable?
    Your big run used about a few million parameters. It is plausible for server/edge GPU deployment, but likely needs pruning, quantization, or a smaller variant for constrained devices.

## Results

25. What metric matters most during development?
    AUROC is the first sanity check because it tells you whether anomalous windows rank above normal windows at all. If AUROC stays near `0.5`, the model is not learning a useful signal regardless of accuracy.

26. Why can accuracy look decent while the detector is still bad?
    The synthetic dataset is class-imbalanced by design, so majority-class accuracy can look fine even when AUROC is near random. The training code now computes `pos_weight` from the dataset and reports score-gap diagnostics to make this visible.

27. What extra diagnostics are available now?
    Validation now logs normal/anomaly score means and the score gap. The attention path also reports hierarchy metrics: p-adic attention correlation, same-cluster attention, different-cluster attention, and hierarchy gap.

28. Is it overfitting?
    Possibly. Training loss steadily decreases while validation loss stays high/noisy.

29. Is the threshold calibrated?
    The training loop now performs validation threshold search and reports best F1, precision, recall, false-positive rate, and threshold. You still need to choose and lock a deployment threshold explicitly.

30. Which metric matters most?
    For deployment, recall and false-positive rate matter most. AUROC is good for research, but an actual detector needs a chosen alert threshold.

## Deployment

31. How would raw events become p-adic tokens?
    You need a mapping from syscall/device event IDs into Hensel digits. Right now, the mapping is synthetic.

32. How often would it run?
    It would run on sliding windows of event streams, for example every new event or every few events.

33. What if normal behavior changes?
    You need retraining, online calibration, or adaptive thresholds.

34. How many false alarms are acceptable?
    That depends on the use case. For security monitoring, low false positives are critical.

35. Can it run without GPU?
    Yes, your CPU path works, but large models will be slower.

36. Can it be quantized/2-adic hardware mapped?
    Your repo already has INT8/2-adic verification code, but the full transformer is not yet hardware-mapped.

37. What should I use for realistic training?
    Turn on `--realistic-dataset` in `scripts/train_anomaly_detector.py`. That path uses idle-heavy windows, low attack rates, and a frequency-weighted loss. Attack injection now excludes idle tokens for the relevant attack types and retries no-op mutations instead of silently labeling unchanged windows as attacks.

## Criticism

37. What baselines are needed?
    Standard transformer without p-adic encoding, LSTM, autoencoder, isolation forest, and random/hash event embeddings.

38. Does p-adic encoding help?
    Not proven yet. The current repo is set up to test that question more cleanly than before because the attention model now mixes learned content logits with a gated p-adic term instead of using p-adic similarity alone.

39. What if p-adic structure is randomized?
    That should be an ablation. If AUROC and hierarchy metrics stay similar after shuffling hierarchy, the p-adic design is not carrying the result.

40. Can attackers evade it?
    Yes, likely. Any anomaly detector can be evaded if attackers learn normal-looking patterns.

41. What convinces a skeptical reviewer?
    Real datasets, baselines, ablations, multiple seeds, threshold analysis, and runtime/deployment measurements.

## Next Experiments

42. Compare `p=3,5,7`?
    You should run the same training config for all three and compare AUROC/F1/runtime.

43. Compare `r=8,16,32`?
    Yes. This tests whether more p-adic precision improves detection.

44. What happens without contrastive loss?
    Run with `--alpha 0.0`. That tells you whether the p-adic contrastive term helps.

45. What happens with a smaller model?
    Try `d_model=128`, fewer layers, and compare AUROC vs speed.

46. How does runtime scale?
    Your `p=7, r=16` big run took `72.7s` for 10 epochs. Repeat with larger `r`, window size, and batch size.

47. How stable across seeds?
    Use `scripts/run_open_dataset.py --seeds 3` or repeat the synthetic training scripts with different `--seed` values and compare AUROC/F1 variance.

## Final 3 Things To Do

1. Start with BCE-first hybrid attention

```bash
make train-attention-bce-gpu
```

2. Compare primes and hierarchy behavior
   Run the same setup for `p=3`, `p=5`, and `p=7`, then compare best AUROC/F1 together with hierarchy correlation and hierarchy gap.

3. Move to realistic training
   After BCE-first synthetic training shows separation, run the realistic path and check whether hierarchy metrics remain meaningful.

## Real Dataset Workflow

Use this when you want to validate on ADFA-LD or BETH instead of synthetic windows.

### ADFA-LD

```bash
make open-adfa
```

Or directly:

```bash
python scripts/run_open_dataset.py --dataset adfa --data-dir ./data/adfa
```

Add `--seeds 3 --quantize-int8` to get a quick stability sweep and CPU INT8 latency comparison.

### BETH

```bash
make open-beth
```

Or directly:

```bash
python scripts/run_open_dataset.py --dataset beth --data-dir ./data/beth
```

Add `--seeds 3 --quantize-int8` for the same benchmarking path on BETH.

### Stats only

```bash
make open-adfa-stats
```

This prints dataset size, real attack rate, `pos_weight`, and the ultrametric verdict without training.

### Realistic training mode

If you want to keep the synthetic generator but make it closer to hardware traces:

```bash
make train-attention-realistic-gpu
```

## Quick Start

```bash
make setup
make train-attention-cpu
```

For the recommended GPU experiment path:

```bash
make train-attention-bce-gpu
```

For the hierarchy and sparsity sweep:

```bash
make sweep-p-bases
```

For the INT8 2-adic hardware dry-lab:

```bash
make int8
```

For a real dataset benchmark:

```bash
make open-adfa
```

## Validation

Run the standard-library tests:

```bash
make test
```

## Current Status

The repo is in a better experimental state than the initial synthetic runs, but it is still not publication-ready.

- The attention model is now hybrid rather than hierarchy-only.
- Synthetic and realistic attack generation now retry no-op mutations instead of silently introducing label noise.
- Training reports threshold-search and score-gap diagnostics.
- The attention path reports hierarchy-alignment metrics directly.

The main open work is still experimental:

- compare against strong non-p-adic baselines
- run multiple seeds
- add randomized-hierarchy ablations
- validate on real datasets with the new metrics

The local reference output is stored in [`results/reference_benchmark.md`](results/reference_benchmark.md).

## Make Targets

- `make open-adfa` downloads and benchmarks ADFA-LD.
- `make open-beth` downloads and benchmarks BETH.
- `make open-adfa-stats` prints ADFA-LD stats without training.
- `make train-attention-bce-gpu` is the recommended first GPU run for the hybrid attention model.
- `make train-attention-realistic-gpu` runs the realistic idle-heavy training path.
- `make sweep-p-bases` reports sparsity plus hierarchy-alignment metrics across `p`.
- `make tnorm-gpu` and `make tnorm-cpu` still run the non-attention baseline path.

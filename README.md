# Personal Project: p-adic Transformer

This repo builds a synthetic p-adic anomaly detection pipeline around Hensel-coded token sequences, a transformer encoder, and a binary anomaly head.

## Big Picture

1. What problem am I solving?
   You are building an anomaly detector for system-event-like sequences. In the current code, it is not detecting real attacks yet; it detects synthetic "attack windows" inserted into Hensel-coded syscall-style sequences.

2. Why ultrametric/tree-like structure?
   Your generator creates token clusters where items in the same class share low-order Hensel prefixes. That means similarity is based on shared p-adic prefix depth, which naturally forms a tree-like hierarchy.

3. What does p-adic distance capture?
   It captures closeness by shared low-order digits, not ordinary geometric distance. Two tokens are close if they share a longer Hensel prefix.

4. Why transformer?
   Your model treats each window as a sequence and uses attention to learn patterns across tokens. This is useful because attacks are sequence-level disruptions, not only single-token anomalies.

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
    It learns sequence patterns over embedded Hensel tokens.

21. What does the anomaly head predict?
    One binary logit per window: normal vs anomalous.

22. Why BCE plus contrastive loss?
    BCE trains the anomaly label. Contrastive loss encourages p-adic structure to separate normal/attack windows.

23. What does contrastive loss encourage?
    Pairs with the same label should be closer; pairs with different labels should be farther apart.

24. Is the model deployable?
    Your big run used about a few million parameters. It is plausible for server/edge GPU deployment, but likely needs pruning, quantization, or a smaller variant for constrained devices.

## Results

25. What does AUROC `0.7681` mean?
    The model has meaningful ranking ability: anomalous windows tend to receive higher anomaly scores than normal windows, but it is not highly reliable yet.

26. Why high accuracy but lower F1?
    The decision threshold is probably not calibrated well, and the dataset may be class-imbalanced.

27. Why best AUROC at epoch 5?
    After epoch 5, the model kept improving some thresholded metrics, but ranking quality peaked. That suggests possible overfitting or calibration drift.

28. Is it overfitting?
    Possibly. Training loss steadily decreases while validation loss stays high/noisy.

29. Is the threshold calibrated?
    No. Your code uses a fixed logit threshold of `0.0`. You should tune the threshold on validation data.

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
    Turn on `--realistic-dataset` in `scripts/train_anomaly_detector.py`. That path uses idle-heavy windows, low attack rates, and a frequency-weighted loss. You can also benchmark the trained model with dynamic INT8 quantization from the open-dataset runner.

## Criticism

37. What baselines are needed?
    Standard transformer without p-adic encoding, LSTM, autoencoder, isolation forest, and random/hash event embeddings.

38. Does p-adic encoding help?
    Not proven yet. You need ablations comparing p-adic vs non-p-adic encodings.

39. What if p-adic structure is randomized?
    That should be an ablation. If performance stays the same, the p-adic design may not be adding much.

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

1. Run ablations

```bash
python scripts/train_anomaly_detector.py --device cuda --p 7 --r 16 --alpha 0.0 --epochs 10 --n-train 131072 --n-val 16384 --batch-size 768
```

2. Compare primes
   Run the same setup for `p=3`, `p=5`, and `p=7`, then compare best AUROC/F1.

3. Tune the threshold
    Right now F1 depends on a fixed threshold. Add validation threshold search and report best F1, recall, precision, and false-positive rate.

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
python scripts/train_anomaly_detector.py --realistic-dataset --realistic-attack-fraction 0.005 --idle-fraction 0.70
```

## Quick Start

```bash
make setup
make train-cpu
```

For a GPU training run:

```bash
make tnorm-gpu
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

The local reference output is stored in [`results/reference_benchmark.md`](results/reference_benchmark.md).

## Make Targets

- `make open-adfa` downloads and benchmarks ADFA-LD.
- `make open-beth` downloads and benchmarks BETH.
- `make open-adfa-stats` prints ADFA-LD stats without training.
- `make tnorm-gpu` runs the normal GPU training pipeline.
- `make tnorm-cpu` runs the normal CPU training pipeline.
- `make train-cpu` remains as an alias for `make tnorm-cpu`.
- `make train-attention-cpu` runs the attention model smoke test.

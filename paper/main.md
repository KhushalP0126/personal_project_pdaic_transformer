# When Do 2-adic Inductive Biases Help?

## Abstract

We study two separate p-adic inductive biases for anomaly detection on Hensel-coded sequences: Hensel digit embeddings and explicit ultrametric attention bias. The controlled matrix compares a raw-token transformer, a flat digit baseline, a Hensel-only transformer, the original sigmoid-gated p-adic attention model, and a signed-alpha attention variant. Across three CPU seeds, signed alpha performs best on the simple synthetic IP task, but it does so while keeping the explicit bias near zero. On the harder transition task, performance falls back toward chance. Under generator shift, transfer is weak. On a realistic idle-heavy proxy, the raw-token and flat-digit baselines outperform the p-adic attention variants on average. The safe conclusion is that structured digit features help on the hierarchy-aligned synthetic task, while explicit ultrametric attention bias is conditional and often safest when it can shut itself off.

## 1. Introduction

This repo no longer supports a strong “new method beats all baselines” claim. The more defensible question is narrower and more useful:

> Is the gain coming from Hensel coordinates, from an explicit p-adic attention kernel, or from both?

That is the framing of the current study.

The IP-prefix task is attractive because hierarchy is concrete: IPv4 addresses share meaningful prefix structure, and a model that decomposes addresses into digits can potentially generalize across unseen raw addresses. But that same property creates a baseline caveat. A raw-token model with a training-only vocabulary is intentionally weak on unseen validation addresses. So the paper needs to distinguish three possibilities:

1. any digitized representation helps
2. Hensel digit-position structure helps beyond flat digit features
3. an additional explicit ultrametric attention bias helps beyond the embedding

This study is built to separate those three.

## 2. Method

### 2.1 Two p-adic mechanisms

There are two separate p-adic mechanisms in the code:

| Mechanism | Variant | Meaning |
|---|---|---|
| Hensel digit embedding | `hensel_only` | p-adic coordinate representation |
| Ultrametric attention bias | `hensel_padic_sigmoid`, `hensel_padic_signed_alpha` | relational hierarchy prior between tokens |

The important point is that these are not the same intervention. A model can benefit from digit coordinates even if the explicit attention bias is useless or harmful.

### 2.2 Attention formula

The attention path is:

```text
content_logits = QK^T / sqrt(d)
padic_logits = normalized p-adic valuation matrix
attention_logits = content_logits + alpha * padic_logits
```

The bias mode determines `alpha`:

| Mode | Alpha behavior | Interpretation |
|---|---|---|
| `none` | `alpha = 0` | Hensel-only, no explicit p-adic bias |
| `sigmoid` | `alpha in [0,1]` | old positive-only gate |
| `signed_alpha` | `alpha = alpha_max * tanh(raw_alpha)` | can attract, ignore, or oppose p-adic closeness |

The signed-alpha change is important because the old gate can only strengthen or weaken a positive prior. It cannot represent “this hierarchy is misleading.”

### 2.3 Model variants

The final study compares:

| Variant | Hensel embedding | Explicit p-adic attention bias | Purpose |
|---|---|---|---|
| `standard_transformer` | no | no | raw-token baseline |
| `flat_digit_transformer` | no | no | tests whether flat digit or bit structure is already enough |
| `hensel_only` | yes | no | isolates Hensel coordinates |
| `hensel_padic_sigmoid` | yes | yes, positive only | old gate |
| `hensel_padic_signed_alpha` | yes | yes, signed | tests whether the model should attract, ignore, or oppose p-adic closeness |

### 2.4 Baseline caveat

The raw-token baseline builds its vocabulary from the training split and maps unseen validation ids to one OOV token. That is deliberate. It tests raw-token generalization without digit sharing. We use it as an inductive-bias comparison, not as a claim that standard Transformers fail in general.

## 3. Experimental design

### 3.1 Tasks

We evaluate five settings:

1. simple synthetic IP-prefix anomalies
2. harder transition-rule synthetic anomalies
3. cross-generator simple->transition
4. cross-generator transition->simple
5. realistic idle-heavy proxy

### 3.2 Budget

All final study numbers were run on CPU with:

- seeds: `20260504`, `20260505`, `20260506`
- epochs: `3`
- `d_model=64`
- `n_layers=1`
- `n_heads=4`
- `train_samples=2048`
- `val_samples=512`

The realistic proxy uses the same train/validation counts with:

- `realistic_window_size=32`
- `realistic_attack_fraction=0.05`
- `realistic_idle_fraction=0.70`

### 3.3 Metrics

Primary metric:

- AUROC

Secondary metrics:

- F1
- precision
- recall
- accuracy

Signed-alpha diagnostics:

- `padic_alpha`
- `raw_padic_alpha`
- `padic_alpha_grad_norm`
- `padic_attention_corr`
- `hierarchy_gap`
- `content_logit_std`
- `padic_logit_std`

## 4. Results

### 4.1 Simple synthetic

Mean AUROC over 3 seeds:

- standard: `0.5000 +- 0.0000`
- flat digit: `0.5372 +- 0.0209`
- hensel-only: `0.5703 +- 0.0634`
- old gate: `0.5811 +- 0.0461`
- signed alpha: `0.6065 +- 0.0191`

This is the one setting where the structured models clearly help. Signed alpha is best, but the learned alpha is still near zero:

- `padic_alpha = -0.0032 +- 0.0004`

So the right interpretation is not “signed alpha learns strong p-adic attraction.” The right interpretation is that signed alpha wins by allowing the explicit p-adic term to become nearly inert when that is the safest solution.

The new flat-digit baseline matters here. It narrows the difference between “digit structure helps” and “specifically Hensel helps.” The Hensel-only model still beats flat digit on average, but the margin is not large enough to justify a strong claim.

### 4.2 Transition synthetic

Mean AUROC:

- standard: `0.5000 +- 0.0000`
- flat digit: `0.4802 +- 0.0136`
- hensel-only: `0.4877 +- 0.0299`
- old gate: `0.4901 +- 0.0219`
- signed alpha: `0.4982 +- 0.0075`

This task removes most of the earlier synthetic advantage. Signed alpha edges the old gate, but everything is effectively near chance. The corresponding alpha remains near zero:

- `padic_alpha = 0.0009 +- 0.0008`

That is strong evidence against the idea that a robust explicit ultrametric bias is doing heavy lifting here.

### 4.3 Cross-generator transfer

Simple->transition:

- standard: `0.5000 +- 0.0000`
- flat digit: `0.4911 +- 0.0287`
- hensel-only: `0.4876 +- 0.0186`
- old gate: `0.5003 +- 0.0402`
- signed alpha: `0.5016 +- 0.0257`

Transition->simple:

- standard: `0.5000 +- 0.0000`
- flat digit: `0.5148 +- 0.0467`
- hensel-only: `0.4744 +- 0.0211`
- old gate: `0.4823 +- 0.0164`
- signed alpha: `0.5065 +- 0.0389`

Transfer is weak. The best row is the flat-digit baseline on transition->simple, and none of the p-adic variants shows strong generator-invariant behavior.

### 4.4 Realistic proxy

Mean AUROC:

- standard: `0.7446 +- 0.0530`
- flat digit: `0.7315 +- 0.0741`
- hensel-only: `0.7186 +- 0.0657`
- old gate: `0.6926 +- 0.0488`
- signed alpha: `0.7100 +- 0.0582`

This is the clearest warning row. The non-p-adic baselines are stronger on average. Signed alpha still beats the old gate, but not because it learns a useful positive hierarchy pull. Again it stays near zero:

- `padic_alpha = -0.0010 +- 0.0001`

The hierarchy diagnostics are negative on average, which is consistent with mismatch between the explicit p-adic prior and this proxy task.

## 5. Discussion

The repo now supports a characterization paper, not a method-superiority paper.

The clean findings are:

1. structured digit features help on the aligned simple synthetic IP task
2. the old positive-only p-adic gate is brittle
3. signed alpha is safer because it can suppress the explicit hierarchy term
4. transfer is weak under generator shift
5. the realistic proxy does not support a broad “p-adic attention helps” claim

The flat-digit baseline made the interpretation stricter in a good way. Before that comparison, it was too easy to read the result as a Hensel-specific or p-adic-specific win. After adding flat digit, the safest story is that structured digit representations help, Hensel may help somewhat more on the simple aligned task, and the explicit ultrametric bias should be treated as conditional.

## 6. Claim boundary

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

## 7. Limitations

The main limitations are unchanged:

- no real routing corpus
- CPU-only budget
- weak transfer
- realistic proxy is still synthetic
- flat digit baseline absorbs part of the story

This means the paper should be positioned as:

> a controlled study of when p-adic inductive bias helps, not a claim of broad p-adic superiority.

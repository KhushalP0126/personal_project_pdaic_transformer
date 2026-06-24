# Outline

## Title

When Do 2-adic Inductive Biases Help? A Controlled Study of Hensel Embeddings and Ultrametric Attention for IP-Prefix Anomaly Detection

## Abstract

We study two separate p-adic inductive biases for anomaly detection on Hensel-coded sequences: Hensel digit embeddings and explicit ultrametric attention bias. The controlled matrix compares a raw-token transformer, a flat digit baseline, a Hensel-only transformer, the original sigmoid-gated p-adic attention model, and a signed-alpha attention variant. Across three CPU seeds, signed alpha performs best on the simple synthetic IP task, but it does so while keeping the explicit bias near zero. On the harder transition task, performance falls back toward chance. Under generator shift, transfer is weak. On a realistic idle-heavy proxy, the raw-token and flat-digit baselines outperform the p-adic attention variants on average. These results suggest that structured digit features help on hierarchy-aligned synthetic tasks, while explicit ultrametric attention bias is conditional and often safest when it can shut itself off.

## 1. Introduction

- Why inductive bias matters for anomaly detection on structured discrete sequences
- Why IP-prefix data is a natural hierarchy-aligned testbed
- Problem with prior framing: embedding and attention bias were confounded
- Contribution: separate coordinate bias from explicit attention bias

## 2. Method

### 2.1 Hensel embedding

- Digit-wise embedding over p-adic / IP digits
- Shared structure across unseen tokens

### 2.2 Explicit p-adic attention bias

- Attention formula:
  - `content_logits = QK^T / sqrt(d)`
  - `padic_logits = normalized p-adic valuation matrix`
  - `attention_logits = content_logits + alpha * padic_logits`
- Old formulation: `alpha = sigmoid(raw_gate)`
- Limitation: only positive scaling, cannot express hierarchy mismatch

### 2.3 Signed-alpha bias

- `alpha = max_alpha * tanh(raw_alpha)`
- Interpretation:
  - `alpha > 0`: hierarchy helps
  - `alpha = 0`: ignore explicit hierarchy bias
  - `alpha < 0`: hierarchy is misleading

### 2.4 Model variants

- `standard_transformer`
- `flat_digit_transformer`
- `hensel_only`
- `hensel_padic_sigmoid`
- `hensel_padic_signed_alpha`

### 2.5 What each variant tests

| Variant | Hensel embedding | Explicit p-adic attention bias | Purpose |
|---|---|---|---|
| `standard_transformer` | no | no | raw-token baseline |
| `flat_digit_transformer` | no | no | tests whether flat digit or bit features already explain the gain |
| `hensel_only` | yes | no | isolates Hensel coordinates |
| `hensel_padic_sigmoid` | yes | yes, positive only | old gate |
| `hensel_padic_signed_alpha` | yes | yes, signed | tests whether the model should attract, ignore, or oppose p-adic closeness |

## 3. Experimental design

### 3.1 Tasks

- simple synthetic IP-prefix anomalies
- transition-rule synthetic IP anomalies
- simple->transition transfer
- transition->simple transfer
- realistic idle-heavy proxy

### 3.2 Metrics

- primary: AUROC
- secondary: F1, precision, recall, accuracy
- diagnostics: `padic_alpha`, `raw_padic_alpha`, `padic_alpha_grad_norm`, `padic_attention_corr`, `hierarchy_gap`, `content_logit_std`, `padic_logit_std`

### 3.3 Baseline caveat

- The raw-token baseline builds its vocabulary from the training split and maps unseen validation ids to one OOV token.
- This is intentional: it tests raw-token generalization without digit sharing.
- We should not interpret weak raw-token performance as a general claim about Transformers.

## 4. Results

### 4.1 Simple synthetic

- signed alpha `0.6065 +- 0.0191`
- old gate `0.5811 +- 0.0461`
- hensel-only `0.5703 +- 0.0634`
- flat digit `0.5372 +- 0.0209`
- standard `0.5000 +- 0.0000`

Takeaway: structure helps on the aligned synthetic task, but the gap between Hensel and flat digit baselines means the paper must separate digit-structure benefit from specifically Hensel benefit.

### 4.2 Transition synthetic

- standard `0.5000 +- 0.0000`
- signed alpha `0.4982 +- 0.0075`
- old gate `0.4901 +- 0.0219`
- hensel-only `0.4877 +- 0.0299`
- flat digit `0.4802 +- 0.0136`

Takeaway: the harder sequential rule removes the synthetic advantage almost entirely.

### 4.3 Cross-generator transfer

- simple->transition: signed alpha `0.5016 +- 0.0257`, old gate `0.5003 +- 0.0402`, standard `0.5000 +- 0.0000`
- transition->simple: flat digit `0.5148 +- 0.0467` is best

Takeaway: generator-specific structure is a real limitation; none of the p-adic variants transfers cleanly.

### 4.4 Realistic proxy

- standard `0.7446 +- 0.0530`
- flat digit `0.7315 +- 0.0741`
- hensel-only `0.7186 +- 0.0657`
- signed alpha `0.7100 +- 0.0582`
- old gate `0.6926 +- 0.0488`

Takeaway: realistic mismatch favors the non-p-adic baselines on average, while signed alpha still beats the old gate by keeping the explicit bias near zero.

## 5. Discussion

- The old positive-only gate is too rigid
- Signed alpha is safer because it can neutralize the explicit hierarchy term
- Flat digit structure is a stronger baseline than the earlier raw-token comparison suggested
- The method is characterization, not superiority proof

## 6. Limitations

- no real routing dataset yet
- small CPU-only training budget
- transfer instability under generator shift
- realistic proxy is still synthetic

## 7. Conclusion

The clean claim is not that p-adic attention beats transformers in general. The clean claim is that Hensel coordinates are useful on hierarchy-aligned IP tasks, while explicit ultrametric attention bias is conditional and often best when it can relax toward zero.

# Outline

## Title

When Do 2-adic Inductive Biases Help? A Controlled Study of Hensel Embeddings and Ultrametric Attention for IP-Prefix Anomaly Detection

## Abstract

We study two separate p-adic inductive biases for anomaly detection on Hensel-coded sequences: Hensel digit embeddings and explicit ultrametric attention bias. The controlled matrix compares a standard transformer, a Hensel-only transformer, the original sigmoid-gated p-adic attention model, and a signed-alpha attention variant. On simple IP-prefix synthetic anomalies, Hensel structure helps and signed alpha performs best. On a harder transition-rule task the advantage shrinks. Under generator shift, transfer is weak. On a realistic idle-heavy proxy, signed alpha matches the standard transformer by collapsing the explicit p-adic bias toward zero, while the old sigmoid gate underperforms. These results suggest that p-adic inductive bias is conditional on hierarchy-task alignment and that Hensel coordinates are more reliable than a fixed positive attention prior.

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

- Old formulation: `content_logits + sigmoid(raw_gate) * padic_logits`
- Limitation: only positive scaling, cannot express hierarchy mismatch

### 2.3 Signed-alpha bias

- New formulation: `content_logits + alpha * padic_logits`
- `alpha = max_alpha * tanh(raw_alpha)`
- Interpretation:
  - `alpha > 0`: hierarchy helps
  - `alpha = 0`: ignore explicit hierarchy bias
  - `alpha < 0`: hierarchy is misleading

### 2.4 Model variants

- `standard_transformer`
- `hensel_only`
- `hensel_padic_sigmoid`
- `hensel_padic_signed_alpha`

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

## 4. Results

### 4.1 Simple synthetic

- signed alpha `0.6260`
- old gate `0.6089`
- hensel-only `0.5846`
- standard `0.5000`

Takeaway: hierarchy-aligned structure helps, and Hensel coordinates already carry a meaningful share of the gain.

### 4.2 Transition synthetic

- signed alpha `0.5252`
- old gate `0.5205`
- standard `0.5000`
- hensel-only `0.4622`

Takeaway: the harder sequential rule weakens the advantage.

### 4.3 Cross-generator transfer

- simple->transition: hensel-only `0.5400` is best
- transition->simple: standard `0.5000` is best; all others degrade

Takeaway: generator-specific structure is a real limitation.

### 4.4 Realistic proxy

- standard `0.7485`
- signed alpha `0.7484`
- hensel-only `0.6895`
- old gate `0.6013`

Takeaway: signed alpha recovers standard performance by effectively zeroing the explicit p-adic bias.

## 5. Discussion

- Hensel embedding is the more stable inductive bias
- Old positive-only gate is too rigid
- Signed alpha is safer because it can neutralize the explicit hierarchy term
- The method is characterization, not superiority proof

## 6. Limitations

- no real routing dataset yet
- small CPU-only training budget
- transfer instability under generator shift
- realistic proxy is still synthetic

## 7. Conclusion

The clean claim is not that p-adic attention beats transformers in general. The clean claim is that Hensel coordinates are useful on hierarchy-aligned IP tasks, while explicit ultrametric attention bias is conditional and often best when it can relax toward zero.

# Architecture Diagram

```text
IPv4 address window
    |
    v
32-bit binary / p-adic digits
    |
    +------------------------------+
    |                              |
    v                              v
flat digit projection         Hensel digit embedding
    |                              |
    |                              +------------------------------+
    |                                                             |
    v                                                             v
content attention logits                                   optional p-adic valuation logits
    |                                                             |
    |                                     alpha = 0 / sigmoid(raw_gate) / alpha_max * tanh(raw_alpha)
    +------------------------------+------------------------------+
                                   |
                                   v
                 attention_logits = content_logits + alpha * padic_logits
                                   |
                                   v
                           transformer encoder
                                   |
                                   v
                              anomaly score
```

## Variant mapping

- `standard_transformer`: raw token ids, no digit sharing
- `flat_digit_transformer`: flat digit projection, no explicit p-adic bias
- `hensel_only`: Hensel digit embedding, `alpha = 0`
- `hensel_padic_sigmoid`: Hensel digit embedding, `alpha in [0, 1]`
- `hensel_padic_signed_alpha`: Hensel digit embedding, signed `alpha`

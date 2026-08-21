# Qwen3.5 Gated DeltaNet updater v1

Predicts stable `key/value/gate/beta` parameters for recurrent layers 4, 8, 12, 16 and 20 from the preceding attention-anchor output, then reconstructs full `16×128×128` states with Qwen's exact transition algebra.

- parameters: 5,647,184;
- optimizer steps: 300;
- held-out full-state MSE ratio vs copy: 0.842777;
- train-only calibrated state-delta alpha: 0.01;
- state-only held-out true-vocabulary KL: 0.005605 copy → 0.003898 learned;
- with the learned convolution updater, strict full-cache KL: 0.040465 stale → 0.028344 learned.

The checkpoint passes exact-shape/formula and mean full-cache injected-logit gates. It does not pass every-prompt KL, free-generation quality, or measured speedup gates; deployment remains disabled.

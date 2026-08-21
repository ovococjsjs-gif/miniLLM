# Qwen3.5 Gated DeltaNet updater v1

Predicts stable `key/value/gate/beta` parameters for recurrent layers 4, 8, 12, 16 and 20 from the preceding attention-anchor output, then reconstructs full `16×128×128` states with Qwen's exact transition algebra.

- parameters: 5,647,184;
- optimizer steps: 300;
- held-out full-state MSE ratio vs copy: 0.842777;
- train-only calibrated state-delta alpha: 0.01;
- stage-1 state-only KL: 0.005605 copy → 0.003898 learned;
- state-only alpha did not hold over all 16 transitions (`1.0277×` copy);
- with the learned convolution updater, strict 16-transition full-cache KL: 0.021935 stale → 0.015856 learned.

The checkpoint passes exact-shape/formula and mean strict full-cache injected-logit gates. It does not pass every-transition KL, free-generation quality, or measured speedup gates; deployment remains disabled.

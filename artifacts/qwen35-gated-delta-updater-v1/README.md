# Qwen3.5 Gated DeltaNet updater v1

First active AIra-Qwen acceleration checkpoint. It predicts stable `key/value/gate/beta` parameters for recurrent layers 4, 8, 12, 16 and 20 from the preceding attention-anchor output, then reconstructs full `16×128×128` states with Qwen's exact transition algebra.

- parameters: 5,647,184;
- optimizer steps: 300;
- held-out full-state MSE ratio vs copy: 0.842777;
- train-only calibrated injection alpha: 0.01;
- held-out true-vocabulary KL: 0.005605 copy → 0.003898 learned;
- held-out argmax preservation: 4/4 learned vs 3/4 copy.

The checkpoint passes full-state shape/formula and mean injected-logit gates. It does **not** pass every-prompt KL, convolution new-row prediction, free-generation quality, or measured speedup gates; deployment remains disabled.

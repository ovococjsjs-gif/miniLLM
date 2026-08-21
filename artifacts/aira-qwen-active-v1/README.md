# AIra-Qwen active v1

Single normalized inference checkpoint combining the accepted Gated Delta recurrent-state and convolution-row predictors.

- parameters: 8,069,185;
- component training: 300 steps per component, each within the local cap;
- normalization: internal per-sample LayerNorm; train-only corpus statistics are hash-bound audit data and are not applied because tiny-corpus z-scoring regressed held-out state MSE;
- strict full-cache train diagnostic prefers alpha 0.05, but the intervention is frozen at the earlier validation-independent alpha 0.01;
- fixed-scorecard 16-transition full-cache KL ratio: 0.722852 versus stale cache;
- oracle argmax preserved: 16/16; improved transitions: 10/16;
- every-transition, autoregressive-generation, physical-skip, and speed gates remain closed.

Run `python scripts/run_aira_qwen_active.py --prepare-only` to verify that data/runtime are ready, or run without flags for fresh component training, packaging, calibration, and replay.

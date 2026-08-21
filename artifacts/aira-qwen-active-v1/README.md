# AIra-Qwen active v1

Single normalized inference checkpoint combining the accepted Gated Delta recurrent-state and convolution-row predictors.

- parameters: 8,069,185;
- component training: 300 steps per component, each within the local cap;
- normalization: internal per-sample LayerNorm; train-only corpus statistics are hash-bound audit data and are not applied because tiny-corpus z-scoring regressed held-out state MSE;
- train-only strict full-cache calibration selected state alpha 0.05;
- 16-transition held-out full-cache KL ratio: 0.829463 versus stale cache;
- oracle argmax preserved: 16/16;
- every-transition, autoregressive-generation, physical-skip, and speed gates remain closed.

Run `python scripts/run_aira_qwen_active.py --prepare-only` to verify that data/runtime are ready, or run without flags for fresh component training, packaging, calibration, and replay.

# AIra-Qwen active pipeline

This document is the single operational contract for current work. Historical proxies remain evidence only; new training starts from `configs/experiments/aira_qwen_active_v1.json`.

## Normalized project layout

The active pipeline now has one source of truth for:

- donor path/hash and pinned llama.cpp revision;
- transition probe and learned replay binaries;
- raw/compact transition corpus;
- train/validation prompt-group split;
- candidate layers `4, 8, 12, 16, 20`;
- state and convolution training step caps;
- normalization contract;
- component and combined checkpoints;
- calibration split and candidates;
- 16-transition native replay;
- gate status and deployment prohibition.

Large exact transition tensors remain reproducible cache data and are not committed. Every compact array, source, binary, checkpoint, and report is hash-bound.

## Feature normalization decision

The 160 train-only layer events produce audited mean/scale hashes. Applying those tiny-corpus z-scores was explicitly tested and rejected: held-out recurrent-state ratio regressed above copy. The accepted model therefore uses its internal per-sample `LayerNorm` on raw event features. Corpus statistics remain audit-only, while the combined checkpoint stores identity mean/scale buffers. This prevents hidden preprocessing drift without forcing an empirically harmful transform.

## Combined runtime checkpoint

`artifacts/aira-qwen-active-v1/cache-updater.pt` combines:

- the 5.65M-parameter Gated Delta parameter predictor;
- the 2.42M-parameter convolution newest-row predictor;
- normalization buffers;
- exact candidate-layer ordering;
- source/config/component hashes.

Legacy two-checkpoint export and combined-checkpoint export were compared byte-for-byte on the same transition and produced identical `AIRASTP2` patches.

## One-command workflow

Prepare everything without optimizing:

```bash
python scripts/run_aira_qwen_active.py --prepare-only
```

This verifies or builds native binaries, verifies the donor and llama.cpp revision, validates 1,920 required tensors and 480 exact shifted rows, and compiles train-only normalization hashes.

Run fresh bounded training and all gates:

```bash
python scripts/run_aira_qwen_active.py
```

The runner performs:

1. state updater training, 300 steps;
2. convolution updater training, 300 steps;
3. deterministic combined-checkpoint packaging;
4. train-only strict full-cache alpha calibration;
5. validation-only 16-transition native replay;
6. one machine-readable status report.

To validate packaging/calibration/replay without retraining:

```bash
python scripts/run_aira_qwen_active.py --reuse-checkpoints
```

## Fixed scorecard

`configs/experiments/aira_qwen_scorecard_v1.json` freezes the 16 transitions, five candidate layers, alpha `0.01`, strict stale recurrent+convolution baseline, and five headline metrics. Diagnostic stage-1, train, projected-state, and oracle-convolution numbers can no longer replace the headline without a scorecard version bump.

## Current normalized checkpoint result

The combined checkpoint is behaviorally identical to the accepted separate components. Strict full-cache train diagnostics prefer alpha `0.05`, but changing the applied intervention moved the held-out headline. The active scorecard therefore freezes the earlier validation-independent alpha `0.01`. On the fixed 16 validation transitions:

- mean learned/full-copy KL ratio: `0.722852`;
- oracle argmax preserved: `16/16`;
- transitions with lower KL: `10/16`;
- mean full-cache gate: passed;
- every-transition gate: failed;
- autoregressive generation: not passed;
- physical skip speedup: not passed;
- deployment: disabled.

The engineering path is now deterministic and reproducible. Remaining model work is better training/data/objectives plus the explicit autoregressive and physical-skip gates; stored-answer routes are not part of the pipeline.

# AIra Qwen3.5 real recurrent-state probe

Status: first real donor-state extraction gate passed on 2026-08-21. This is an instrumentation result, not a state-patcher or acceleration claim.

## Why this gate exists

Qwen3.5-0.8B is hybrid: layers 0–2, 4–6, 8–10, 12–14, 16–18 and 20–22 are Gated DeltaNet recurrent layers, while layers 3, 7, 11, 15, 19 and 23 are full-attention anchors. Skipping recurrent work without advancing its cache changes all future tokens. The earlier synthetic state-patcher proxy could validate loss plumbing, but it could not prove access to the donor’s real recurrent dynamics.

The pinned llama.cpp public context API exposes `ggml_backend_sched_eval_callback`. The new `native/qwen35_state_probe.cpp` uses that callback without patching llama.cpp itself. It captures graph tensors that the pinned Qwen implementation already names:

- `state_predelta-N` — recurrent state entering layer N;
- `new_state-N` — recurrent state after the Gated DeltaNet update;
- `conv_states-N` and `last_conv_states-N` — convolution cache before and after the event;
- `l_out-N`/`h_pre_norm` — all 24 layer outputs;
- the complete 248,320-element output-logit vector after every captured stage.

Non-contiguous convolution views are gathered into logical row-major float32 arrays before writing. Raw captures stay under ignored `data/`; only hashes, dimensions, dynamics and gate conclusions are committed.

## Measured smoke

Pinned inputs:

- donor SHA-256: `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`;
- llama.cpp revision: `9a532ae4bab1b164052ce60a738f78538b421c66`;
- prompt: 15 tokens;
- continuation: one greedy token;
- stages: prompt completion state plus one autoregressive update.

Results:

| item | measured result |
|---|---:|
| recurrent layers per stage | 18 |
| attention-anchor outputs per stage | 6, within 24 total layer outputs |
| recurrent state per layer | `128×128×16 = 262,144` float32 values |
| recurrent state bytes per layer | 1,048,576 |
| recurrent state bundle per stage | 18,874,368 bytes |
| convolution state per layer | 18,432 float32 values |
| tensor captures | 192, 96 per stage |
| raw smoke size including logs/logits | 84,433,365 bytes |
| full future-logit vectors | 2 × 248,320 float32 |
| recurrent cache transition equality | 18/18 exact |
| convolution cache transition equality | 18/18 exact |

For every recurrent layer, stage-0 `new_state` is byte-identical to stage-1 `state_predelta`. The logically gathered stage-0 `last_conv_states` is likewise byte-identical to stage-1 `conv_states`. This proves that the captured tensors are the live cache chain, not similarly shaped diagnostics.

The first autoregressive token changed the recurrent states by a mean `||Δ|| / ||state_after||` of `0.205614`; the maximum was `0.611461`. Mean before/after cosine was `0.970720`, ranging from `0.801761` to `0.996503`. A skipped event therefore cannot safely leave the state unchanged, even though many layer states remain directionally close.

Machine-readable evidence is in:

- `results/qwen35_state_probe_build.json`;
- `results/qwen35_08b_real_state_probe.json`.

## Reproduction

Build the pinned llama.cpp runtime first, then compile the probe:

```bash
python scripts/bootstrap_qwen35_donor.py --build-runtime --source github
python scripts/build_qwen35_state_probe.py
```

Run and audit an exact two-stage smoke:

```bash
python scripts/run_qwen35_state_probe.py --overwrite
```

The runner verifies the model and probe binary hashes, writes into a temporary directory, atomically publishes raw captures, independently audits dimensions and cache continuity, and then writes the compact committed report.

## What this enables—and what it does not

Now enabled:

1. compilation of real `(state_before, event features, emitted span, state_after)` pairs;
2. exact attention-anchor hidden outputs;
3. complete future-logit supervision;
4. direct zero-delta, copy-state and learned-patcher controls;
5. deterministic replay checks through cache hashes.

Still not demonstrated:

- reconstruction of the 18 full recurrent states after an omitted event;
- acceptable future KL from patched states;
- calibrated fallback confidence;
- generated-quality parity;
- any wall-clock or active-block acceleration.

The next bounded experiment must project and batch multiple real transitions, train the patcher on those projections, and reject it unless held-out future behavior improves over the zero-delta baseline. Full-state low-rank reconstruction and actual skipped-layer execution come only after that control passes.

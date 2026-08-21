# Qwen3.5 projected real-state pairs v1

Compact prompt-grouped learnability controls compiled from exact recurrent states of the pinned Qwen3.5 donor.

- records: 48 (32 train, 16 validation);
- exact cache links: 864/864 recurrent and 864/864 convolution;
- per-layer projected state: 80 float32 values;
- future distribution: 256 probability buckets.

The projections are lossy and cannot be injected as full Qwen states. See `manifest.json` and `docs/aira-qwen35-real-state-probe.md`.

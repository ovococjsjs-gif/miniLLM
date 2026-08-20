# Reference inference and checkpoint evaluation

The PyTorch path is a correctness baseline, not the final phone runtime. It exists so a
training checkpoint can immediately be loaded, generated from, and evaluated before any
export work.

## Generation

```bash
minillm generate runs/pilot/step-1000.pt \
  --tokenizer runs/tokenizers/pilot-32k.json \
  --prompt 'Столица Франции —' \
  --max-new-tokens 32 --json
```

Defaults are deterministic greedy decoding. Sampling is explicit and seeded:

```bash
minillm generate CHECKPOINT --tokenizer TOKENIZER \
  --prompt-file prompt.txt --temperature 0.8 --top-k 40 --top-p 0.95 --seed 123
```

Checkpoint and external config must match exactly. The tokenizer vocabulary must equal
the model vocabulary. PyTorch checkpoints use pickle internally and therefore only
trusted files may be loaded.

## Cache semantics

`MiniLLM.forward_cached` keeps one state per **effective** layer application:

- unexpanded GQA keys/values;
- at most `kernel_size - 1` projected convolution values;
- fixed GDN2 matrix state.

This also handles shared core repetitions by assigning independent temporal state to
each application of the shared weights. Tests compare mixed prefill/chunk/token decode to
full-forward logits. Changing recurrence count invalidates an existing cache.

Engram currently uses safe full-prefix fallback because both suffix-token hashing and its
causal refinement convolution need dedicated cache state. This preserves output semantics
but is slow; `used_cache` in generation reports the actual path.

## Bilingual completion smoke suite

```bash
PYTHONPATH=src python scripts/evaluate_completions.py CHECKPOINT \
  --tokenizer TOKENIZER \
  --suite eval/bilingual_smoke.json \
  --output results/completion_smoke.json
```

The suite covers RU/EN factual continuation, arithmetic, translation, code, and honest
uncertainty. Its transparent substring checks are diagnostics, not an aggregate quality
benchmark. The prompts must be included in the protected decontamination set before
corpus v1 is packed.

## Compact inference checkpoint

`save_inference_checkpoint` writes model weights, architecture config, step, and optional
metadata without optimizer/RNG state. Training checkpoints remain the source for exact
resume; compact checkpoints are for trusted local inference and export only.

# L1: first real-data model package

L1 separates a cheap architecture screen from the first full pass over the pinned GitHub
pilot. Neither is presented as an assistant-quality model.

## Matched controls

| pair | quality control | edge control | parameter difference |
|---|---|---|---:|
| CPU screen | `l1_screen_attention_5m.json` — 4.858M | `l1_screen_edge_5m.json` — 4.861M | 0.069% |
| GPU L1 | `l1_attention_20m.json` — 19.597M | `l1_edge_20m.json` — 19.605M | 0.043% |

Both pairs use the frozen 8K tokenizer, tied embeddings, the same FFN width, depth, MTP
objective and training data. `n_kv_heads / n_heads = 1/2` makes attention and convolution
projection budgets nearly equal. The edge controls replace five of eight or six of ten
attention mixers with causal convolution; they do not change the data or optimizer.

GDN2, MoE, shared depth and Engram are intentionally absent from this gate.

## Preregistered CPU screen

The contract is `configs/experiments/l1_real_screen.json`:

```bash
PYTHONPATH=src python scripts/run_l1_screen.py
```

It is hard-capped at 300 steps. Each run sees 76,800 real tokens; three seeds are required.
The gate checks finite optimization, validation improvement and a maximum 5% edge-quality
regression. It is an implementation/ranking signal, not a final architecture decision.

## Full 20M plan

A dry run requires no GPU:

```bash
PYTHONPATH=src python scripts/train_l1.py --dry-run
```

The default one-pass plan is:

- 19.60M parameters;
- 31,094,503 target tokens;
- batch 8 × sequence 512 × accumulation 8 = 32,768 tokens/optimizer step;
- 949 optimizer steps;
- BF16 autocast, gradient checkpointing and fused AdamW on CUDA;
- token-based 1% warmup and cosine decay;
- exact checkpoint resume plus compact best-inference export.

Actual GPU launch:

```bash
PYTHONPATH=src python scripts/train_l1.py \
  --model configs/l1_attention_20m.json \
  --device cuda --precision bf16 \
  --output runs/l1-attention-20m
```

The edge control changes only `--model configs/l1_edge_20m.json`. A run must first measure
throughput/VRAM and may be stopped early if validation does not improve. The current 31M
unique-token corpus provides only about 1.6 tokens per model parameter; this is a scaling
checkpoint, not enough training for useful language generation.

## Trainer guarantees

Checkpoint format v3 adds:

- FP32/BF16/FP16 autocast and FP16 gradient-scaler state;
- optional block-level gradient checkpointing;
- optional fused AdamW;
- token-based warmup/decay independent of microbatch shape;
- non-finite loss/gradient death guards;
- corpus, token-stream, tokenizer and git metadata checks;
- peak CUDA memory and throughput accounting.

Format-v2 FP32 checkpoints remain resumable after default-field migration. Packed token
sidecars provide content SHA-256, so an identical regenerated stream is accepted even if
its filesystem timestamp changed.

## Success criteria after a full pass

- no non-finite values or late validation reversal;
- loss improvement continues beyond the 300-step screen;
- deterministic completion suite produces non-degenerate text;
- quality/edge difference is interpreted jointly with throughput and KV footprint;
- no claim of assistant usefulness until data reaches at least the planned 0.1B+ range
  and instruction/tool post-training is performed.

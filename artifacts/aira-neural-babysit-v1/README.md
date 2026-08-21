# AIra Neural Babysit v1

This artifact contains a learned prompt-gated hidden-to-logit residual, not a SkillShelf or stored-answer route. The frozen Qwen donor supplies 1024-wide hidden states; 264,137 adapter parameters were optimized for exactly 300 steps.

- Training records: 48
- Held-out free-generation concept passes: 1/24 -> 14/24 before manual review
- Out-of-scope answers preserved: 6/8
- Production deployment: blocked

`adapter.bin` is the native numeric checkpoint; `model.pt` preserves the PyTorch tensors; TSV/JSONL files bind training, held-out prompts, outputs, and provenance. Teacher answers are not runtime inputs. See `docs/aira-neural-babysit-v1.md` and the manually audited result for limits.

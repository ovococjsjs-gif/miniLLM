# AIra Mentor Tiny v1 — interaction smoke only

Random-initialized 1,715,456-parameter MiniLLM trained for exactly 300 assistant-only SFT steps on AIra Mentor v1.

This checkpoint is **not a base model or useful general assistant**. It learned response templates but usually copies incorrect numbers, identifiers and citations. Proper validation perplexity fell to 2.43, yet only 1 of 10 category demonstrations passed strict generated-output verification. Its purpose is plumbing and producing real student failures for the first AI Babysit dataset.

- Checkpoint: `model.pt`
- SHA-256: `d5f0eaa9b98b3a0451f1ee5564bec0c067d9818d348bc4a7c3dc9fe8121576cd`
- Config: `configs/aira_mentor_tiny.json`
- Tokenizer: `artifacts/tokenizer-github-pilot-v1/tokenizer.json`
- Training report: `results/aira_mentor_tiny_training.json`
- Babysit failures: `artifacts/aira-mentor-babysit-v1/`

Use only with trusted local checkpoint loading:

```bash
minillm generate artifacts/aira-mentor-tiny-v1/model.pt \
  --tokenizer artifacts/tokenizer-github-pilot-v1/tokenizer.json \
  --prompt 'Вычисли 12 умножить на 7.' --max-new-tokens 48 --json
```

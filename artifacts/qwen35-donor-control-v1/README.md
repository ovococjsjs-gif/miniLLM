# Qwen3.5 donor protocol-control experiment

A matched 20-task intervention over the fresh seed-46 donor rollout. `aira-protocol-v1` appends category-specific, answer-free source/protocol reminders to the task system message. It does not inspect or insert the deterministic reference answer.

## Matched result

| component | baseline | controlled | delta |
|---|---:|---:|---:|
| strict | 3/20 | 3/20 | 0 |
| content | 12/20 | 13/20 | +1 |
| protocol | 17/20 | 18/20 | +1 |
| required source | 0/7 | 3/7 | +3 |

The control repaired the surface schema for both tool tasks and inserted source IDs in three source-required tasks. It still produced wrong tool arguments, degenerated into a citation list on one memory task, and improved no strict verdict. Therefore this prompt is **not accepted as a quality intervention**. It is evidence that citations and JSON framing are steerable, but deterministic tool execution, selective source pointers and exact content binding must be architectural/runtime actions rather than prompt-only claims.

## Files

- `evaluation.json` — complete controlled generations and current verifier components;
- `comparison.json` — matched per-component transitions against `../qwen35-donor-babysit-v1/evaluation.json`.

Reproduce the comparison:

```bash
python scripts/compare_qwen_donor_evaluations.py \
  --baseline artifacts/qwen35-donor-babysit-v1/evaluation.json \
  --candidate artifacts/qwen35-donor-control-v1/evaluation.json \
  --output artifacts/qwen35-donor-control-v1/comparison.json
```

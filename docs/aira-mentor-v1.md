# AIra Mentor v1 — deterministic verifier-first SFT seed

Date: 2026-08-21. Artifact: [`artifacts/aira-mentor-v1/`](../artifacts/aira-mentor-v1/).

## Purpose

AIra Mentor v1 is a small project-owned RU/EN capability-SFT seed and a bootstrap source for later AI Babysit rollouts. It is deliberately **not** named after or designed to imitate a proprietary model. It contains no hidden chain-of-thought and is not a replacement for base pretraining.

The goal is to establish a clean floor of behavior that is unusually relevant to AIra:

- short verifiable reasoning;
- deterministic tools;
- editable memory and provenance;
- retrieval grounding;
- prompt-injection resistance;
- calibrated unknown answers;
- critique and correction.

## Size and identity

- 6,000 conversations;
- 5,673 train / 153 validation / 174 protected test;
- 3,000 Russian / 3,000 English;
- 10 balanced categories, 600 records each;
- 23 verifier-backed template families;
- 707,593 tokens under the current 8K pilot tokenizer;
- 2,222,943 UTF-8 bytes of rendered chat text;
- exact conversation duplicates: 0;
- one deterministic regeneration retry was needed to avoid a collision;
- corpus SHA-256: `acee70afdcd5e9e8170f635b1c3b710d13cec12a63b28c52f3761e0978327afc`;
- generator SHA-256: `9a8291f3ba20d379387355ee729852a221e00d4fbc760556876377f8731071aa`;
- license: CC0-1.0.

## Categories

| Category | Records | Verification |
|---|---:|---|
| arithmetic | 600 | exact integer arithmetic |
| algebra | 600 | generated solution and substitution identity |
| logic | 600 | deterministic ordering constraints |
| Python | 600 | AST-valid project templates plus generated tests |
| tool calls | 600 | exact calculator/calendar/memory JSON |
| memory control | 600 | known, unknown and conflict policy |
| grounded QA | 600 | exact synthetic document fact and citation |
| prompt injection | 600 | ignored document instruction plus exact citation |
| uncertainty | 600 | absent-field/unknown behavior |
| critique and revision | 600 | independently computed correction of wrong answer |

Each category is balanced 50/50 RU and EN. Records contain generator version, seed, template identity, deterministic split group and machine-readable verification.

## Format

```json
{
  "id": "aira-mentor-v1-tool_call-00000",
  "category": "tool_call",
  "language": "ru",
  "difficulty": "medium",
  "split_group": "tool_call:calculator-json:0",
  "split": "train",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "verification": {
    "kind": "json_equal",
    "expected": {"tool": "calculator", "arguments": {"expression": "..."}},
    "tool_result": 3313,
    "verified": true
  },
  "provenance": {
    "generator": "minillm.aira.synthetic",
    "generator_version": 1,
    "seed": 42,
    "template": "calculator-json",
    "teacher": "deterministic-project-owned"
  },
  "content_sha256": "..."
}
```

Recommended SFT policy is assistant-only loss. The test file must not be used for training, prompt generation, teacher demonstrations or threshold tuning.

## What this dataset can and cannot do

It can teach a small pretrained model consistent local behaviors and provide deterministic regression tests. It cannot supply broad world knowledge, natural dialogue diversity or Gemma-level language ability. At about 0.7M tokens it is intentionally a seed, not a final SFT mixture.

Template generation also has an obvious ceiling. Even with 23 families and randomized values, overtraining will teach templates rather than general intelligence. Use one or a few epochs, track per-category validation, and stop if held-out gains reverse.

## Planned v2 growth through AI Babysit

Do not produce v2 by merely increasing every template from 600 to 6,000. Grow it with new verified task families and on-policy failures:

1. train or load a base checkpoint;
2. run it on fresh tasks generated from protected seeds/licensed documents;
3. execute deterministic verifiers;
4. capture the exact student failure and first-error location;
5. obtain a legally usable mentor correction/distribution;
6. store SFT, preference, process and calibration targets;
7. preserve failing cases in a replay suite;
8. add only examples that improve held-out behavior.

A reasonable next target is 10–20K high-quality records and 3–8M tokens, but only if new families and real on-policy corrections account for most of the growth.

## Reproduction

```bash
python scripts/build_aira_mentor_dataset.py --overwrite
pytest tests/test_aira_synthetic.py
```

The build refuses to overwrite a non-empty artifact unless `--overwrite` is passed. Manifest file hashes bind every split to the generator and tokenizer revision.

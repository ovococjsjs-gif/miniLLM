# Candidate synthetic SFT audit and AI Babysit design

Date: 2026-08-21. These datasets are candidates for later instruction/process tuning, **not** the base pretraining corpus. Dataset-level licenses are uploader claims and do not cure upstream model-service or source-prompt restrictions.

## Verdict table

| Candidate | Pinned revision | Declared license | Verdict | Potential role |
|---|---|---|---|---|
| `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k` | `f0330e0ca46469b3928adef18c2b55f9476d6bd3` | Apache-2.0 | quarantine | small English capability-SFT audit only, pending service-terms clearance and manual verification |
| `Roman1111111/claude-opus-4.6-10000x` | `d6fe6aafcf5db8141153a0828c791eeee512b171` | MIT | reject for base; quarantine for math-SFT research | independently verified arithmetic subset only, pending upstream rights and benchmark-origin reconstruction |
| `WithinUsAI/claude_mythos_distilled_25k` | `2c5e638c51a22b8b883def51bab685ae7e282c72` | Apache-2.0 | reject | templating/slop detector negative set, not training targets |

## 1. Angrygiraffe Opus 4.6/4.7 8.7K

Primary files at the pinned revision:

- `full_train.jsonl`: 8,706 examples, 71,446,661 bytes, LFS SHA-256 `6d9fd97e13da19cdb86f6ffd6876285e4a21d9c8a371cc25e0f9368dbb61c6c8`;
- `full_train_no_reasoning.jsonl`: 48,355,573 bytes, LFS SHA-256 `30860240458408001e48371e2009da400062b65097e508fe5ffb6ba38b73de7a`;
- overlapping instruct/roleplay/code subsets and corresponding no-reasoning variants.

The Hugging Face default viewer exposes about 38.5K rows because it combines overlapping full and subset files plus reasoning/no-reasoning variants. Loading the default split would therefore duplicate examples. Any audit must select exactly one canonical file.

Positive signals:

- broad categories, multi-turn conversations and varied system prompts;
- explicit model tag per row;
- reasoning and no-reasoning variants;
- plausible size (~17M claimed tokens for the reasoning file).

Blocking issues:

- the card explicitly says the dataset was not manually reviewed;
- its `<think>` content is synthetic visible reasoning, not Claude's hidden chain of thought, despite contradictory marketing language calling it “genuine”;
- generated through a Claude Max consumer plan. Anthropic Consumer Terms prohibit using the service to develop/train AI models, and prohibit automated/non-human access except where explicitly permitted;
- roleplay deliberately imitates named authors and includes a “no refusals or safety hedging” design goal;
- English-only and unsuitable as base knowledge data;
- medical, legal, finance and factual answers need independent checking.

Do not use the overlapping default config. If legal clearance is ever obtained, begin with a manually sampled `instruct_train_no_reasoning.jsonl`, remove roleplay, validate code/math with tools, and cap its SFT mixture share.

## 2. Roman Opus 4.6 10K

Pinned file:

- `opus46_final.jsonl`: 9,633 rows, 13,404,840 bytes, LFS SHA-256 `f2282f16d0fc225e03b6f0485b9eb5eef1a0b9ff55157a13c02cb6a7eefa5ea2`.

Positive signals:

- compact `messages` plus separate assistant `reasoning` field;
- many arithmetic answers can be independently recomputed;
- simple format and explicit claimed teacher model.

Blocking issues found in direct samples:

- most early data is elementary GSM8K-like arithmetic, not frontier reasoning;
- later samples include unrelated summarization, movie comprehension and low-quality casual prompts;
- metadata categories are visibly wrong (movie comprehension marked `math`, news summary marked `code`);
- at least one sampled document contains mojibake/encoding damage;
- the card claims 27.2M input+output tokens while the stored file is only 13.4 MB; this likely includes generation traffic not retained training content and must not be quoted as stored tokens;
- source prompts appear drawn from public benchmarks/corpora without per-row origin/license, creating protected-evaluation contamination;
- actual Claude output raises the same upstream service-terms issue; an uploader MIT label cannot grant rights the uploader did not have.

Not suitable for base pretraining. A future research-only path could reconstruct source provenance, remove protected eval sets, independently solve each arithmetic task, discard the supplied reasoning, and keep only verified prompt/final-answer pairs after legal review. That work is likely more expensive than generating a clean owned set.

## 3. WithinUsAI Mythos Distilled 25K

Pinned file:

- `claude_mythos_distilled_25k.jsonl`: 25,000 rows, 55,191,861 bytes, LFS SHA-256 `5e94ca487977c43f1c5c5dfda06b1d4a9ba4ac6591bc69b48202c7b44455be51`.

The card itself says this is **not** real Claude/Mythos output; it is an independent synthetic mirror generated from another model's interpretation. Direct rows show strong templates and immediate repetition: BLAKE3 and allocator prompts recur within the first few examples, prompts repeatedly request “Mythos-style” or “complete production-ready” treatment, and responses use a repeated branded opening. Some prompts request impossible or underspecified deliverables while demanding invented metrics/formal verification.

Additional provenance inconsistency: the card says a reproducible generator script is included, but the pinned repository tree contains only `.gitattributes`, `README.md` and the JSONL file—no generator.

The mixture is English-only and heavily skewed toward cybersecurity (7K/25K). This is imitation-of-imitation style data, not capability distillation. Using it risks teaching verbosity, fabricated benchmarks, false authority and a repeated persona. Reject as training targets; retain small samples only as a detector/evaluation set for synthetic slop and unsupported claims.

## Upstream terms risk

Anthropic's current Consumer Terms prohibit using Claude services to develop or train AI/ML models and prohibit automated/non-human access except through allowed interfaces. Current Commercial Terms likewise prohibit accessing the services to build a competing product, including training competing AI models, without express approval. Output ownership language is conditioned on compliance with those terms.

Therefore a public repository's Apache/MIT label is not enough by itself for Claude-generated traces. Production inclusion requires evidence of express permission or a legal determination that the intended downstream use is allowed. Until then these candidates remain outside production and research weight builds.

## AI Babysit: the owned alternative

The proposed “older sibling” loop is technically sound. It is a combination of on-policy distillation, DAgger/expert iteration, process supervision, critique-and-revision and AI feedback. It directly addresses the failure observed in AIra: the student leaves the corpus manifold, after which corpus targets and shelf calibration no longer apply.

### Correct loop

1. Sample a task from an owned/licensed task generator.
2. Freeze the exact student checkpoint and let it produce an answer/action trace.
3. Run deterministic verification first: unit tests, calculator, schema, retrieval citations, memory provenance.
4. A mentor evaluates the **exact student state**, locates the first error and produces a correction/rubric.
5. Store original answer, verifier observations, critique, corrected answer and sparse teacher distribution with teacher/student hashes.
6. Train several separate objectives:
   - SFT on corrected answers;
   - preference loss with corrected answer chosen over original;
   - step/process loss at the first error;
   - Top-K KL on exact student-generated prefixes;
   - calibrated abstention/uncertainty targets.
7. Re-run held-out tasks and retain regressions in a replay buffer.
8. Promote a checkpoint only after capability, safety and generated-context gates all pass.

The model does not learn merely because feedback is written in natural language. Feedback must be converted into optimizer targets, and the next rollout must come from the updated checkpoint.

### Mentor record

```json
{
  "task_id": "stable-task-id",
  "student_checkpoint": "sha256",
  "teacher_id": "model-or-human-revision",
  "prompt": "...",
  "student_answer": "...",
  "verifier_observations": [{"tool": "pytest", "passed": false, "detail": "..."}],
  "verdict": "incorrect",
  "first_error_offset": 127,
  "error_type": "invalid_assumption",
  "critique": "...",
  "corrected_answer": "...",
  "rubric_scores": {"correctness": 0, "grounding": 1, "style": 2},
  "teacher_confidence": 0.94,
  "constitution_flags": []
}
```

### Task sources

Prefer tasks that are grounded and verifiable:

- arithmetic/programmatic math with executable answers;
- code with unit tests and static analysis;
- retrieval QA generated from licensed documents with exact citations;
- JSON/tool protocols with schema validators;
- explicit-memory insert/update/conflict/delete tasks;
- bilingual translation/extraction with parallel licensed sources;
- adversarial uncertainty/unknown tasks where abstention is correct.

Free-form synthetic “write an expert answer about anything” should be a minority and receive human spot checks. A small 5K set of diverse, verified, on-policy corrections is more valuable than 25K repetitive imitation answers.

### Teacher rights

Use an open-weight teacher with a compatible license, a teacher service that explicitly permits model training/distillation, or project-owned/human labels. Do not assume that outputs from any hosted assistant—including an AI acting as the babysitter—are legally reusable for training without checking that service's terms.

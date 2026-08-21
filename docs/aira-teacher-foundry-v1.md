# AIra Teacher Foundry v1

Status: implemented and reproduced on 2026-08-21.

## Purpose

Teacher Foundry turns a bounded number of high-quality teacher interventions into an executable, verifier-backed curriculum. The teacher is the Arena.ai coding agent working with the project owner; small local donor models are not treated as authorities.

The design deliberately avoids bulk imitation chat. One teacher review identifies a failure mechanism, specifies an algorithm and verifier, and lets deterministic generators produce many fresh instances:

```text
student rollout
→ deterministic observations
→ failure fingerprints
→ cause clusters
→ teacher packet
→ skill patches
→ generated + on-policy contrastive curriculum
→ protected evaluation
```

Teacher examples contain concise visible rationales and operational procedures, not purported hidden chain-of-thought.

## Implemented records

`src/minillm/aira/foundry.py` provides:

- `FailureFingerprint`: category, verifier and inferred causal failure mode;
- `TeacherCluster`: all task IDs plus a bounded set of representative failures;
- `TeacherPacket`: a hash-bound review packet for one student checkpoint;
- `SkillPatch`: diagnosis, executable algorithm, verifier/tool/uncertainty policy and RU/EN contrastive examples;
- `CurriculumRecord`: chosen/rejected answers with source and patch identity;
- deterministic packet, patch and curriculum writers with collision checks.

The first patch catalog contains 11 patches:

1. numeric operand ledger;
2. equation isolation plus substitution;
3. ordered constraint chains;
4. Python contract-first implementation;
5. tool argument grounding;
6. exact memory key/source resolution;
7. grounded field/citation binding;
8. untrusted-document authority boundaries;
9. missing-evidence abstention;
10. independent critique recomputation;
11. response-state isolation for cross-task contamination.

`src/minillm/aira/verification.py` is now the shared strict verifier used by tiny training, Babysit collection, donor evaluation and Foundry compilation. The Python verifier no longer accepts code merely because it compiles: it admits a small AST subset, denies imports/I/O/unbounded loops/arbitrary calls, executes deterministic tests and verifies input non-mutation.

## Reproduced v1 artifact

Command:

```bash
python scripts/build_aira_teacher_foundry.py
```

Inputs:

- 200 seed-43 tiny-student rollouts;
- checkpoint SHA-256 `d5f0eaa9b98b3a0451f1ee5564bec0c067d9818d348bc4a7c3dc9fe8121576cd`;
- 193 failed and 7 accepted responses;
- fresh deterministic seed 44, separate from Mentor v1 and the seed-43 rollout.

Outputs under `artifacts/aira-teacher-foundry-v1/`:

| file | role |
|---|---|
| `teacher-packet.json` | 11 failure clusters with at most three representatives each |
| `skill-patches.json` | the teacher-authored patch catalog |
| `curriculum.jsonl` | 1,193 preference/SFT-ready records |
| `curriculum.jsonl.manifest.json` | source, patch, category and SHA-256 manifest |
| `report.json` | compact build report |

Curriculum composition:

- 1,000 fresh deterministic generated contrastive records, 100 per category;
- 193 exact on-policy student corrections;
- no protected Mentor v1 train/validation/test record was consumed;
- zero identifier or content collisions;
- curriculum SHA-256 `972f34c559618feb7c6682917ede6ad7c5323474f54bf47827884b7249a795ec` (all 6,000 Mentor v1 content hashes excluded during fresh generation).

Observed failure mechanisms:

- 59 arithmetic/algebra/critique operand-binding failures;
- 1 explicit cross-task contamination failure;
- 20 constraint/entity failures;
- 20 Python symbol/constant failures;
- 20 stale tool-argument failures;
- 13 memory source-binding failures;
- 60 document-source binding failures across grounding, injection and uncertainty.

This confirms that the tiny model primarily learned response surfaces while failing to bind current prompt values.

## Training use

Every `CurriculumRecord` contains ordinary assistant-only `messages` as well as `chosen` and `rejected` answers. It can therefore support:

- assistant-only SFT on `chosen`;
- preference objectives on `(chosen, rejected)`;
- patch-balanced sampling;
- failure-mode-specific evaluation;
- later process/action supervision from the patch algorithm.

It is not a base-pretraining corpus. Repeating it into a random-init tiny model is not expected to create general language capability.

## Matched tiny-model intervention

`finetune_aira_mentor_foundry.py` continued the published 1.715M-parameter tiny checkpoint for exactly 300 additional optimizer steps on the 1,193-record curriculum. After explicit exclusion of every Mentor v1 content hash, the corrected matched rerun improved perplexity on the unchanged 153-record validation split from `2.427031` to `2.222130`. On the same ten fresh seed-45 generated tasks, however, strict verification remained `0/10 → 0/10`. The output still substituted stale numbers, identifiers, document IDs and tool arguments.

This is a useful negative control: Foundry makes corrections denser and auditable, but cannot manufacture the missing language/binding capacity in a random-init 1.7M model. The resulting checkpoint is retained only as a local experiment and is not published as a better assistant. The machine-readable report is `results/aira_mentor_tiny_foundry_finetune.json`.

## Provenance and release status

The 1,000 generated task targets and all 193 corrections are deterministic project records. Skill-patch language and 22 contrastive demonstrations were authored by the Arena.ai agent under user direction. Internal experiments are approved by the project owner; terms must be reviewed before publicly releasing weights trained specifically on agent-authored text. The artifact records this status rather than incorrectly labeling all content CC0 or attributing it to a named proprietary model.

## Qwen donor Foundry cycle

The recovered Qwen donor was run on a balanced protected sample and 20 fresh seed-46 tasks. The fresh rollout produced 17 corrections: 7 source failures, 5 content failures, 3 combined content/protocol failures and 2 strict-surface failures. The protected 50-task answers remain evaluation-only.

`artifacts/aira-teacher-foundry-qwen-v1/` now compiles those attempts into:

- 17 exact on-policy corrections plus 3 preserved passes in the source packet;
- 11 causal clusters, including operand drift, source identity, memory source/conflict, Python contract, tool schema and tool argument binding;
- the existing 11-patch catalog, with no unjustified new patch;
- 1,000 fresh deterministic seed-47 contrastive examples;
- 1,017 total curriculum records with SHA-256 `7e28525b0955efa63b763ae3a2ec6a43d32f2da5db3ec1514a21006eaa885de8`;
- zero protected Mentor v1 records in the weight-producing manifest.

Next:

1. train AIra routes/state patching on generated spans rather than repeating random-init tiny SFT;
2. implement selective source pointers and deterministic tool execution as runtime actions;
3. evaluate every intervention against the frozen protected sample;
4. add a new patch only when a fresh failure cannot be explained by this catalog.

# AIra Teacher Foundry — Qwen donor cycle v1

The second on-policy Foundry cycle, compiled from the 20 fresh seed-46 Qwen donor attempts. The open checkpoint remains the student/donor; corrections come from deterministic AIra references and verifier observations.

## Inventory

- source attempts: 20;
- accepted attempts: 3;
- corrected attempts: 17;
- causal failure clusters: 11;
- Skill Patches: 11;
- deterministic seed-47 contrastive tasks: 1,000;
- exact on-policy correction records: 17;
- total curriculum records: 1,017;
- protected Mentor v1 split records used: 0.

The failure packet distinguishes operand drift, source identity, memory source/conflict, Python contract, tool schema and tool argument binding instead of grouping every strict failure together.

## Files

- `teacher-packet.json` — bounded representatives and 11 clustered mechanisms;
- `skill-patches.json` — assistant-authored algorithms, policies and examples;
- `curriculum.jsonl` — fresh contrastive tasks plus exact donor corrections;
- `curriculum.jsonl.manifest.json` — provenance, category/patch counts and hashes;
- `report.json` — compact build summary.

Curriculum SHA-256: `7e28525b0955efa63b763ae3a2ec6a43d32f2da5db3ec1514a21006eaa885de8`. All 6,000 Mentor v1 content hashes were excluded during fresh generation.

This is a post-base capability curriculum, not a base-pretraining replacement. Public weight use of agent-authored patch text still requires the recorded Arena terms review.

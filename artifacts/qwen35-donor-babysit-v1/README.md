# Qwen3.5 donor Babysit v1

A bounded on-policy diagnostic set for the pinned Qwen3.5-0.8B Q4 donor. These are 20 fresh deterministic seed-46 tasks (two per AIra Mentor category), not records copied from the protected Mentor v1 splits.

## Files

- `tasks.jsonl` — exact fresh prompts and deterministic references;
- `evaluation.json` — raw donor generations plus strict/component verdicts;
- `records.jsonl` — AI Babysit records with exact student answers, verifier observations, critiques and corrections;
- `records.jsonl.manifest.json` — provenance and content hashes;
- `report.json` — compact outcome and failure-cluster counts.

## Result

- strict: 3/20;
- corrected: 17/20;
- failure clusters: source 7, content 5, content+protocol 3, strict-surface 2;
- task SHA-256: `ec6b156b129803e23eae986888f7c8bff1b3ea5b3a4c182e5d06a9b1eaa9505a`;
- record SHA-256: `a8be5d0c87f256c5bb2cfee41beae818b04a56fb83c8bfa35fcb5cb74fdf21e3`.

The collector independently re-runs the current verifier, validates generation hashes, rejects protected content by default and never treats this donor as the teacher.

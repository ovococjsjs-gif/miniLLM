# AIra Mentor v1

Deterministic, project-owned, verifier-first RU/EN seed dataset for later capability SFT and AI Babysit bootstrapping.

- Records: 6,000
- Train/validation/test: 5,673/153/174
- Languages: RU 3,000, EN 3,000
- Categories: 10 × 600
- Verifier-backed template families: 23
- License: CC0-1.0
- Generator seed: 42
- Generator SHA-256: `9a8291f3ba20d379387355ee729852a221e00d4fbc760556876377f8731071aa`
- Corpus SHA-256: `acee70afdcd5e9e8170f635b1c3b710d13cec12a63b28c52f3761e0978327afc`

This is **not** base pretraining data, not real or imitated Claude/Opus output, and contains no hidden chain-of-thought. It teaches compact verified behavior in arithmetic, algebra, logic, Python, calculator JSON, explicit memory, grounded QA, prompt-injection resistance, uncertainty and critique/revision.

All assistant targets are generated from deterministic state and carry machine-readable verification metadata. Code targets are executed against generated tests during construction. Use assistant-only loss and keep the protected test split out of training and teacher generation.

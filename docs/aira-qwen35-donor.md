# Qwen3.5-0.8B language donor control

Status: exact checkpoint recovered, audited, served and evaluated on 2026-08-21.

## Role

Qwen3.5-0.8B is a pretrained language donor and matched control, not AIra's teacher and not the final architecture. Competency corrections come from Teacher Foundry, deterministic tools and verifiers. The donor contributes pretrained RU/EN representation, grammar and a structurally useful hybrid starting point.

The model has 24 layers arranged as six groups of three Gated DeltaNet blocks followed by one full-attention block. That structure is materially closer to the AIra state/event program than Gemma E4B, while the Q4 artifact remains inside the approved package budget.

## Recovery from the main-branch transfer

The user uploaded a 7-Zip split archive as 50 ordinary Git blobs on `main`:

- parts `001`–`049`: 10,485,760 bytes each;
- part `050`: 5,407,258 bytes;
- concatenated archive: 519,209,498 bytes;
- archive SHA-256: `c96703cd1631f2d5849903142fe0088c7427fbbf09c5785d350a1c5ccda5ef7f`.

The session branch remained unchanged. The parts were read directly from the `main` Git tree, concatenated in numeric order and extracted outside Git. The resulting file is:

```text
data/external/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M-github.gguf
```

Verification succeeded exactly:

- size: 532,517,120 bytes;
- SHA-256: `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`;
- exact match to the pinned `unsloth/Qwen3.5-0.8B-GGUF` Q4_K_M object.

The model remains ignored runtime data. The reconstructed transfer archive was removed after its hash and extracted payload were validated. The 50 chunks are intentionally not merged into the Arena branch, so this work does not add another 519 MB to its patchset.

## Embedded GGUF audit

`results/qwen35_08b_model_audit.json` records the parsed metadata:

- GGUF V3, architecture `qwen35`;
- quantized by Unsloth;
- embedded license `apache-2.0`;
- 752,393,024 parameter elements;
- 320 tensors: 133 F32, 98 Q4_K, 36 Q5_K, 36 Q8_0 and 17 Q6_K;
- 24 blocks, hidden size 1024, FFN size 3584;
- 8 attention heads and 2 KV heads;
- full-attention interval 4, hence 18 Gated DeltaNet and 6 attention layers;
- recurrent state size 128, 16 groups, inner size 2048 and convolution width 4;
- native context metadata 262,144 tokens.

The recovered GGUF has no separately named MTP tensors. Upstream MTP training must not be confused with an MTP payload present in this particular artifact.

## Runtime result

A CPU-only llama.cpp `b9222` build at revision `9a532ae4bab1b164052ce60a738f78538b421c66` loaded the model successfully. With two CPU threads, context 2048, batch 128 and greedy non-thinking generation:

| metric | result |
|---|---:|
| model file | 532.5 MB |
| child peak RSS | 852,180 KiB |
| process wall time for bounded smoke | 3.151 s |
| prompt throughput | 79.5 token/s |
| generation throughput | 21.4 token/s |

The RU arithmetic smoke correctly produced `17 + 8 - 6 = 19`. Full machine-readable evidence is in `results/qwen35_08b_runtime_smoke.json`.

## Generated quality baseline

The protected test was sampled at five records per each of ten categories, for 50 balanced tasks. Generation was greedy, non-thinking and capped at 256 tokens.

Strict result:

- `6/50`, or 12%;
- algebra `5/5`;
- Python `1/5`;
- every other category `0/5` under the strict contract.

Diagnostic components explain the gap:

| component | pass |
|---|---:|
| core answer content | 31/50 (62%) |
| response protocol | 40/50 (80%) |
| required source attribution | 0/18 (0%) |
| complete strict contract | 6/50 (12%) |

Examples show that the model often finds a grounded value, queue position or missing-evidence decision but omits the required document/memory source. It also fails exact tool-call protocol and often writes a computed result instead of the requested tool action. Arithmetic composition is unreliable: the first protected task omitted the number of machines. These diagnostic checks never replace strict acceptance.

`results/qwen35_08b_donor_baseline.json` contains all generated answers, hashes, latency, component checks and category totals.

## Fresh Babysit rollout

Twenty new seed-46 tasks, two per category, were generated outside all Mentor v1 splits:

- strict `3/20` (15%);
- content `12/20` (60%);
- protocol `17/20` (85%);
- required source attribution `0/7`;
- 17 exact verifier-backed corrections.

The resulting diagnostic/on-policy data is in `artifacts/qwen35-donor-babysit-v1/`. Failure summary:

- 7 source failures;
- 5 content failures;
- 3 combined content/protocol failures;
- 2 strict-surface failures despite acceptable content;
- 3 accepted generations.

This empirically confirms the design decision: Qwen is useful pretrained tissue, but it is not a competent teacher. Teacher Foundry and deterministic verifiers remain authoritative.

## Matched protocol-control intervention

A category-specific but answer-free `aira-protocol-v1` system suffix was tested on the same 20 fresh tasks. It reminded the donor about exact source IDs, tool JSON shape, Python fencing and final-answer forms without reading the reference answer.

| component | baseline | controlled | delta |
|---|---:|---:|---:|
| strict | 3/20 | 3/20 | 0 |
| content | 12/20 | 13/20 | +1 |
| protocol | 17/20 | 18/20 | +1 |
| required source | 0/7 | 3/7 | +3 |

The intervention fixed the surface schema of both tool calls but not their arguments. It inserted source IDs in three cases, while one memory answer collapsed into a long citation list. No strict verdict changed. The prompt is therefore rejected as a quality fix: it shows that surface citation/protocol behavior is steerable, but content binding, selective pointers and deterministic tool execution must be implemented as verified runtime/architecture actions. Full generations and matched transitions are in `artifacts/qwen35-donor-control-v1/`.

## Reproduction

Build and verify the runtime/artifact:

```bash
.venv/bin/pip install -e '.[runtime]'
python scripts/bootstrap_qwen35_donor.py --build-runtime --source github
```

Run the server:

```bash
.cache/llama.cpp/build/bin/llama-server \
  --model data/external/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M-github.gguf \
  --alias qwen3.5-0.8b-q4-k-m \
  --ctx-size 2048 --threads 2 --threads-batch 2 \
  --batch-size 128 --ubatch-size 128 \
  --parallel 1 --cache-ram 0 \
  --reasoning off --host 127.0.0.1 --port 8080
```

Run a bounded balanced evaluation rather than an unbounded overnight command:

```bash
python scripts/evaluate_qwen35_donor.py \
  --endpoint http://127.0.0.1:8080 \
  --dataset artifacts/aira-mentor-v1/test.jsonl \
  --examples-per-category 5 --max-tokens 256

# Answer-free matched protocol control on a fresh task file
python scripts/evaluate_qwen35_donor.py \
  --endpoint http://127.0.0.1:8080 \
  --dataset artifacts/qwen35-donor-babysit-v1/tasks.jsonl \
  --control-profile aira-protocol-v1 --max-tokens 256 \
  --output artifacts/qwen35-donor-control-v1/evaluation.json
```

`OpenAIChatProvider` is a reusable standard-library adapter and the evaluator deliberately loads its stdlib-only verification helpers without importing Torch.

## Architecture experiment

The first real AIra surgery is not arbitrary layer deletion:

1. retain six attention groups as quality anchors;
2. add calibrated fast/balanced/deep exits;
3. allow shelf/tool/memory events to emit spans without full donor execution;
4. update skipped recurrent states with a learned `RecurrentStatePatcher`;
5. compare future-window logits and generated quality against a full donor pass;
6. fall back to the full group whenever patch confidence is below a generated-context threshold.

The 200-step synthetic dynamics proxy reduced held-out state MSE from a zero-delta baseline of `0.058125` to `0.003531` (ratio `0.06075`), reduced future KL from `0.014753` to `0.000780`, and kept anchor max error exactly zero. This validates only the patcher plumbing.

The instrumentation half of the real-state gate is now complete. A public llama.cpp evaluation callback captured all 18 live `128×128×16` Gated DeltaNet states, their convolution caches, all 24 layer outputs and complete future logits. Prompt-state outputs matched the next autoregressive pre-states byte-for-byte in all 18 recurrent layers, as did all 18 convolution caches. See `docs/aira-qwen35-real-state-probe.md`. Training and validating the patcher on projected real-state pairs remains the next gate; no acceleration claim follows from extraction alone.

## Acceptance gate

No acceleration claim is allowed until:

- AIra quality regression is no more than two absolute percentage points from this generated baseline;
- first-gate active block compute falls by at least 50%;
- state confidence is calibrated on held-out generated events;
- wall-clock and RSS improve, not just theoretical block count;
- failures continue to enter Teacher Foundry rather than being silently accepted.

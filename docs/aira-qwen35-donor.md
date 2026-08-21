# Qwen3.5-0.8B language donor control

Status: runtime prepared; checkpoint download blocked by the current sandbox network on 2026-08-21.

## Role

Qwen3.5-0.8B is a pretrained language donor and matched control, not AIra's teacher and not the final architecture. Competency corrections come from Teacher Foundry, deterministic tools and verifiers. The donor contributes pretrained RU/EN representation, grammar and a structurally useful hybrid starting point.

The pinned upstream card describes a 24-layer, hidden-size-1024 model arranged as six groups of three Gated DeltaNet blocks followed by one gated-attention block, with a 248,320-token tied embedding/output table and trained MTP. This is materially closer to the AIra state/event program than Gemma E4B, while remaining small enough for the approved artifact budget.

Canonical metadata is frozen in `configs/donors/qwen35_08b.json`:

- upstream: `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`;
- quant repository: `bartowski/Qwen_Qwen3.5-0.8B-GGUF` at `f36b1ea49a332ede8fe5f389bbf5b3575ef71f48`;
- file: `Qwen_Qwen3.5-0.8B-Q4_K_M.gguf`;
- exact size: 579,615,840 bytes;
- expected SHA-256: `fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526`;
- license metadata: Apache-2.0;
- llama.cpp: release `b9222`, revision `9a532ae4bab1b164052ce60a738f78538b421c66`.

## Prepared local path

A CPU-only llama.cpp `b9222` build completed successfully in `.cache/llama.cpp`; its reported revision matches the frozen config. Build products and upstream weights are deliberately excluded from Git.

## GitHub mirror audit

An exact GitHub Git LFS mirror of Unsloth's Q4_K_M was found and pinned:

- repository: `leonardomathon/lepa`;
- commit: `3d1afc7b8496435f4a751402bbff240103c30820`;
- path: `models/llm/Qwen3.5-0.8B-Q4_K_M.gguf`;
- size: 532,517,120 bytes;
- Git LFS/file SHA-256: `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`.

The hash exactly equals the Q4_K_M SHA-256 published by `unsloth/Qwen3.5-0.8B-GGUF`; it is not merely a similarly named file. Two independent GitHub repositories, `hrithiks2019/tmp_storage` and `charly-chrtx/maestro`, contain pointers to the same object and size. A different `bopalvelut-prog/qwen3.5-gguf` file was audited but not selected because it points to a different 529,297,312-byte `diodel` quant.

The mirror repository does not expose a machine-readable license even though the upstream model and embedded GGUF metadata declare Apache-2.0. It is therefore accepted only as a hash-verified internal baseline source; public redistribution still uses the upstream license/provenance review.

The bootstrap script resolves and validates the Git LFS pointer itself before accepting any downloaded bytes:

```bash
python scripts/bootstrap_qwen35_donor.py --source github --download
```

Direct GitHub file page: [Qwen3.5-0.8B-Q4_K_M.gguf](https://github.com/leonardomathon/lepa/blob/3d1afc7b8496435f4a751402bbff240103c30820/models/llm/Qwen3.5-0.8B-Q4_K_M.gguf).

In this sandbox, `github.com`, its API and `codeload.github.com` are reachable, but Git LFS redirects the actual object to `github-cloud.githubusercontent.com`; TLS is terminated on that storage hostname. Hugging Face and Xet binary hosts fail in the same way. `results/qwen35_donor_bootstrap.json` therefore records:

- runtime: verified;
- GitHub pointer/revision/hash: verified before transfer;
- model artifact: still missing locally;
- failing storage host: `github-cloud.githubusercontent.com`;
- baseline evaluation: not run.

This is an environment egress blocker, not a model-memory failure and not a benchmark result. The GitHub mirror should download normally outside this sandbox, and a manually downloaded file placed at the configured path is always re-hashed before use.

## Bootstrap and verification

Build the pinned runtime and prefer the GitHub mirror:

```bash
.venv/bin/pip install -e '.[runtime]'
python scripts/bootstrap_qwen35_donor.py \
  --build-runtime --source github --download
```

The original Bartowski/Hugging Face candidate remains available with `--source huggingface`. Each source has its own frozen revision, byte size and SHA-256; the script never treats the two different quant files as interchangeable.

A manually transferred GitHub mirror file must be placed at:

```text
data/external/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M-github.gguf
```

Then run the local server with a bounded context:

```bash
.cache/llama.cpp/build/bin/llama-server \
  --model data/external/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M-github.gguf \
  --alias qwen3.5-0.8b-q4-k-m \
  --ctx-size 2048 --threads 2 --batch-size 128 \
  --host 127.0.0.1 --port 8080
```

And evaluate generated behavior, not teacher-forced likelihood:

```bash
python scripts/evaluate_qwen35_donor.py \
  --endpoint http://127.0.0.1:8080 \
  --dataset artifacts/aira-mentor-v1/test.jsonl
```

`OpenAIChatProvider` in `src/minillm/aira/provider.py` is a reusable standard-library adapter for this endpoint. Credentials, if a remote compatible endpoint is ever used, are read only from an environment variable and are never serialized.

## Architecture experiment

The first real AIra surgery is not arbitrary layer deletion:

1. retain six attention groups as quality anchors;
2. add calibrated fast/balanced/deep exits;
3. allow shelf/tool/memory events to emit spans without full donor execution;
4. update skipped recurrent states with a learned `RecurrentStatePatcher`;
5. compare future-window logits and generated quality against a full donor pass;
6. fall back to the full group whenever patch confidence is below a generated-context threshold.

`src/minillm/aira/state_patcher.py` implements the generic masked state update and objective. Its loss combines state MSE, state cosine distance, future-window KL and confidence calibration. Unpatched anchor layers are exactly preserved.

The 200-step synthetic dynamics proxy in `results/aira_state_patcher_proxy.json` reduced held-out state MSE from a zero-delta baseline of `0.058125` to `0.003531` (ratio `0.06075`), reduced future KL from `0.014753` to `0.000780`, and kept anchor max error exactly zero. This proves the state-patcher training path and masks; it does **not** prove Qwen state reconstruction. Donor hidden-state pairs remain the next required test.

## Acceptance gate

No acceleration claim is allowed until all of the following are measured on the same generated suite:

- full donor baseline completed;
- AIra quality regression no more than two absolute percentage points;
- at least 50% active block-compute reduction in the first gate;
- state confidence calibrated on held-out generated events;
- wall-clock and RSS improve, not just theoretical block count;
- failures are fed back into Teacher Foundry rather than silently routed as successful.

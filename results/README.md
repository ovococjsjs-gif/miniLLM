# Results

Небольшие machine-readable результаты воспроизводимых proxy experiments.

- `toy_mqar.json` — первый generated architecture screen;
- `tokenizer_proxy.json` — multilingual byte-BPE 4K/8K/16K/32K;
- `lm_ablation.json` — 300-step real-text comparison, 2 variants × 3 seeds;
- `lm_hybrid_followup.json` — single-seed 3/4-attention placement check;
- `proxy_forward_benchmark.json` — short reference CPU forward timings;
- `l1_real_screen.json` — preregistered 5M Attention/Edge real-data screen;
- `l1_attention_20m_kaggle.json` — completed 31M-token Attention run on Kaggle P100;
- `sequence_mixer_benchmark.json` — attention/conv/GDN2 time and state scaling;
- `adaptive_depth_proxy.json` — R=1/2/4 MQAR speed/quality and depth consistency;
- `pointer_chase_depth_proxy.json` — iterative-composition curriculum stress test;
- `aira_trigger_proxy.json` — frozen EN/RU byte/char shelf frontier and burst metrics;
- `aira_calibration_proxy.json` — separate calibration/test precision-controlled routing;
- `aira_autonomous_proxy.json` — autonomous 64-token shelf bursts with oracle fallback;
- `aira_memory_proxy.json` — bounded random-code recall, rejection, scan work and latency;
- `aira_residual_proxy.json` — matched full/hard/soft 300-step residual controls;
- `aira_token_bridge_proxy.json` — negative control: raw shelf only at BPE-token boundaries;
- `aira_byte_event_proxy.json` — dynamic-BPE byte-event core, proper loss, calls, traffic and autonomous generation;
- `aira_byte_event_broad_proxy.json` — same 300-step core with an 8M-token unique sampling window;
- `aira_event_core_ablation.json` — matched MLP/conv/attention and random/contiguous/noise/recovery controls (18 runs);
- `aira_event_packing_proxy.json` — lossless multi-byte/shelf/prompt-copy compression upper bounds;
- `aira_event_training_smoke.json` — prepared-shard, multi-head loss, checkpoint and resume plumbing smoke;
- `aira_mentor_tiny_training.json` — 1.7M random-init, 300-step SFT smoke and strict generated failures;
- `aira_mentor_tiny_foundry_finetune.json` — matched 300-step Foundry intervention: lower PPL but unchanged 0/10 fresh strict passes;
- `aira_state_patcher_proxy.json` — 200-step synthetic state catch-up, future-KL and exact-anchor control;
- `qwen35_donor_bootstrap.json` — exact recovered donor and pinned llama.cpp verification;
- `qwen35_08b_model_audit.json` — embedded GGUF architecture, license, tensor and quantization inventory;
- `qwen35_08b_runtime_smoke.json` — bounded 2-thread RSS and prompt/generation throughput;
- `qwen35_state_probe_build.json` — pinned llama.cpp/public-API native probe build provenance;
- `qwen35_08b_real_state_probe.json` — exact 18-layer recurrent/conv cache chains, hidden outputs and full logits;
- `qwen35_real_state_patcher_proxy.json` — 48-pair projected real-state control versus copy and mean-delta baselines;
- `qwen35_08b_donor_baseline.json` — balanced 50-task protected strict/content/protocol/source baseline;
- `pc_alm_proxy.json` — old finite PC versus PC-ALM BP-gradient alignment.

Каноническая интерпретация AIra-результатов находится в
[`docs/aira-v2-audit.md`](../docs/aira-v2-audit.md). Эти файлы не являются target-model quality claims. Полный контекст и ограничения описаны в
[`docs/toy-mqar-result.md`](../docs/toy-mqar-result.md) и
[`docs/small-scale-experiments.md`](../docs/small-scale-experiments.md).

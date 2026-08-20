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
- `pc_alm_proxy.json` — old finite PC versus PC-ALM BP-gradient alignment.

Каноническая интерпретация AIra-результатов находится в
[`docs/aira-v2-audit.md`](../docs/aira-v2-audit.md). Эти файлы не являются target-model quality claims. Полный контекст и ограничения описаны в
[`docs/toy-mqar-result.md`](../docs/toy-mqar-result.md) и
[`docs/small-scale-experiments.md`](../docs/small-scale-experiments.md).

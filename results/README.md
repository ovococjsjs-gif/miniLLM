# Results

Небольшие machine-readable результаты воспроизводимых proxy experiments.

- `toy_mqar.json` — первый generated architecture screen;
- `tokenizer_proxy.json` — multilingual byte-BPE 4K/8K/16K/32K;
- `lm_ablation.json` — 300-step real-text comparison, 2 variants × 3 seeds;
- `lm_hybrid_followup.json` — single-seed 3/4-attention placement check;
- `proxy_forward_benchmark.json` — short reference CPU forward timings;
- `l1_real_screen.json` — preregistered 5M Attention/Edge real-data screen;
- `l1_attention_20m_kaggle.json` — completed 31M-token Attention run on Kaggle P100.

Эти файлы не являются target-model quality claims. Полный контекст и ограничения описаны в
[`docs/toy-mqar-result.md`](../docs/toy-mqar-result.md).

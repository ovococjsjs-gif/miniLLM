# Быстрые proxy-результаты в текущей среде

Текущий режим: отдельный training test ограничивается **не более чем 300 steps**. Эти
эксперименты отбраковывают плохие идеи и проверяют plumbing; они не заменяют масштабирование.

## 1. Multilingual tokenizer proxy

Corpus: 1,189 документов / 19.7 MB из Universal Dependencies: English EWT, German GSD,
Russian SynTagRus и Ukrainian IU. Train selection ограничен 100 документами на язык,
чтобы большой Russian treebank не задавил остальные. Russian/Ukrainian источники имеют
NC-лицензию и используются только для research proxy, не как production training corpus.

| Vocab | Q4 tied embedding при d=1024 | bytes/token | tokens/word | Оценка decode MFLOP/UTF-8 byte |
|---:|---:|---:|---:|---:|
| 4K | 2 MiB | 4.396 | 2.691 | 150.8 |
| 8K | 4 MiB | 5.000 | 2.366 | 134.3 |
| 16K | 8 MiB | 5.616 | 2.107 | 122.5 |
| 32K | 16 MiB | 6.235 | 1.897 | **115.7** |

Для русского `tokens/word` падает с 2.744 при 4K до 1.934 при 32K; для украинского —
с 2.984 до 2.098. На этом proxy 32K vocabulary окупает LM-head рост сокращением числа
autoregressive steps. Это поддерживает target 32K–65K, но финальный выбор требует гораздо
большего web/code/dialogue corpus.

Полные данные: [`results/tokenizer_proxy.json`](../results/tokenizer_proxy.json).

## 2. Real-text LM ablation, 300 steps

Две почти iso-parameter модели (около 1.74M), 3 seeds, одинаковые 153,600 tokens/run,
sequence 128, MTP weight 0.2. LM token stream использует весь train split и поэтому
сильно смещён в русский из-за размера SynTagRus; это architecture smoke test, а не
сбалансированная multilingual quality оценка:

| Variant | Attention / conv | Mean best validation main loss | Std | Mean wall time |
|---|---:|---:|---:|---:|
| Edge hybrid | 2 / 4 | 6.726 | 0.041 | 31.3 s |
| Attention-only | 6 / 0 | **6.586** | 0.057 | 34.0 s |

На коротком реальном тексте attention-only заметно лучше по loss, а edge hybrid примерно
на 8% быстрее в полном training run. Single-seed follow-up дал loss 6.757 для 3 attention
и 6.738 для 4 attention — они пока не превосходят крайние варианты.

Reference PyTorch CPU forward, batch 4 × sequence 128:

| Attention / conv | Median forward | tokens/s |
|---:|---:|---:|
| 2 / 4 | **15.59 ms** | **32,832** |
| 3 / 3 | 16.52 ms | 30,999 |
| 4 / 2 | 17.36 ms | 29,501 |
| 6 / 0 | 18.01 ms | 28,424 |

Итого: каждая замена attention на convolution здесь даёт скорость, но отнимает LM quality.
Пока сохраняются два baseline: `quality=attention-only` и `edge=2 attention + 4 conv`.
Выбор нельзя делать по одному aggregate score: на generated MQAR edge hybrid ранее получил
97.4%, а attention-only 28.7%, то есть inductive bias сильно зависит от задачи.

Полные данные:

- [`results/lm_ablation.json`](../results/lm_ablation.json);
- [`results/lm_hybrid_followup.json`](../results/lm_hybrid_followup.json);
- [`results/proxy_forward_benchmark.json`](../results/proxy_forward_benchmark.json).

## 3. Support-aware n-gram draft shelf

На полном 4K byte-BPE train split (3.70M tokens) построены continuation counts порядков
2/4/8; проверка — отдельный validation split (428.8K tokens). Заранее заданный primary
gate с нижней границей Wilson ≥0.90 получил **2.523% coverage при 98.188% accuracy** и
прошёл контракт. Агрессивный empirical probability ≥0.90 при support 4 получил больше
coverage (6.683%), но лишь 93.784% accuracy и непригоден для bypass.

Результат поддерживает только opt-in speculative draft с neural verification. Он не
доказывает ускорение: требуется multi-token verifier и end-to-end device benchmark.

- [`configs/experiments/ngram_draft_proxy.json`](../configs/experiments/ngram_draft_proxy.json);
- [`results/ngram_draft_proxy.json`](../results/ngram_draft_proxy.json).

## 4. Active-byte decode-energy proxy

При placeholder LPDDR 60 pJ/byte, MAC 0.5 pJ и контексте 8K Fermi estimate составляет
13.562 mJ/token для dense 350M, 13.776 для recursive 200M, 8.079 для MoE-кандидата и
10.924 для hybrid GDN2 (включая read+write 6.75 MiB state). Это прозрачная статическая
оценка, не measured watts: irregular dispatch, cache residency и kernels могут изменить
ранжирование.

Полные компоненты: [`results/decode_energy_proxy.json`](../results/decode_energy_proxy.json).

## 5. Вывод для следующего цикла

- Не увеличивать steps, а расширять eval axes.
- Сравнить layer ratios на generated retrieval/state tracking и short real-text loss.
- Добавить QAT speed/quality smoke test.
- Следующий архитектурный test: shared-depth при 1/2/3 loops с одинаковым максимумом 300 steps.
- Mobile conclusion принимать только после ExecuTorch/llama.cpp microbenchmark.

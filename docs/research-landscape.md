# Карта исследований: что действительно полезно для маленькой модели

Дата среза: **20 августа 2026**. Приоритет отдаётся статьям, официальным model cards
и измерениям на устройствах. Маркетинговые сравнения не считаются доказательством.

## 1. Ограничения задачи

Цель не сводится к минимуму параметров. Для локального ассистента одновременно важны:

- RAM модели и cache;
- TTFT, prefill и decode при batch=1;
- энергия, нагрев и throttling после нескольких минут;
- качество на русском и английском;
- instruction following, retrieval и tool use;
- устойчивость, калибровка и честное «не знаю»;
- возможность управлять памятью и поведением.

Поэтому «активных параметров мало» ещё не означает «быстро на телефоне»: MoE может
упереться в нерегулярный доступ к весам, а linear attention — в отсутствующий kernel.

## 2. Наиболее сильные подтверждённые направления

| Направление | Что показано | Что переносим | Ограничение доказательства |
|---|---|---|---|
| MobileLLM | На 125M/350M deep-thin, tied embeddings, GQA и попарное sharing улучшают качество | глубина, GQA, tied embeddings, weight reuse | старые модели, 1T tokens, без современных hybrid-блоков |
| LFM2 | Hardware-in-the-loop search выбрал 10 gated short-conv + 6 GQA для 350M–1.2B; добавление SSM/linear attention ухудшало device Pareto | простой conv/GQA baseline обязан быть первым | конкретные CPU, runtime и закрытая часть training data |
| MobileMoE | 0.3–0.9B active; sweet spot — умеренная sparsity, fine-grained и shared experts; на телефонах 2.2–3.4× decode speedup против dense baseline | MoE как вторая ветка после custom operator | работа от 26 мая 2026, нужен grouped/custom kernel |
| Kimi Linear | 3:1 KDA:MLA, до 75% меньше KV и до 6.3× decode throughput на 1M context | hybrid linear/global attention для длинных траекторий | проверено на 48B-A3B, не на sub-billion ARM CPU |
| Gated DeltaNet-2 | Раздельные channel-wise erase/write gates лучше KDA и Mamba-3 в matched 1.3B/100B setup | более сильный recurrent-memory кандидат, чем копирование KDA | очень новая работа; reference kernel сложнее обычного conv |
| Recurrent depth / MoR | Повторение общих весов даёт test-time latent compute; MoR экономит параметры и динамически выбирает глубину | «думать дольше» без загрузки больших весов | нестабильность обучения и сложный dynamic batching |
| Engram | 20–25% sparse capacity выгодно отдать deterministic n-gram memory; O(1) lookup | вынести статические шаблоны из neural compute | масштаб 27B; неизвестна оптимальная доля на 200–350M |
| BitNet b1.58 | 2B модель занимает около 0.4 GB non-embedding memory и быстро работает на CPU с bitnet.cpp | отдельная native-ternary ветка | нужен pretraining from scratch и специальные kernels |
| BLT | Dynamic byte patches сравнялись с BPE на крупных масштабах и экономили до 50% inference FLOPs | tokenizer/patching — полноценная архитектурная ось | сложнее runtime; выигрыш для 200–350M не доказан |

## 3. DeepSeek V4 Flash: правильный ориентир, но не маленькая модель

DeepSeek-V4-Flash — **284B total / 13B active**, а не телефонная модель. Официальная
архитектура сочетает CSA и HCA, DeepSeekMoE, mHC, MTP и Muon. CSA сжимает каждые
4 KV entry и выбирает top-k, HCA сжимает каждые 128 и использует dense attention.
Это объясняет дешёвый миллионный контекст, но не даёт готового рецепта для ARM.

Полезные переносимые принципы:

1. **Сжимать вдоль sequence, а не только feature dimension.**
2. **Хранить локальные детали отдельно от далёкого compressed context.**
3. **MTP одновременно улучшает supervision и даёт draft tokens.**
4. **mHC показывает, что residual topology — самостоятельная ось scaling.**
5. **Muon следует честно сравнить с AdamW по loss на wall-clock, не по шагам.**
6. **Engram отделяет lookup от computation — это особенно логично для маленькой модели.**

Не переносим в v0: HCA/CSA, четыре residual streams и сотни experts. Их operator
complexity не оправдана до появления сильного простого baseline.

## 4. KDA, GDN2 или просто convolution?

У KDA сильные результаты на длинном контексте: channel-wise decay, fixed recurrent
state и периодический global attention. GDN2 улучшает формулу, разделяя erase и write:

```text
S̄_t = D_t S_{t-1}
r_t = S̄_tᵀ (b_t ⊙ k_t)
S_t = S̄_t + k_t (w_t ⊙ v_t - r_t)ᵀ
```

Но LFM2 сообщает важный отрицательный результат: в hardware-in-the-loop поиске для
edge CPU дешёвые gated convolutions плюс шесть GQA-блоков дали лучший aggregate
quality/latency Pareto, а SSM/linear-attention варианты не улучшили его. Поэтому порядок
работы такой:

1. fused conv/GQA baseline;
2. GDN2/KDA только в matched параметрах, данных и wall-clock;
3. оставить recurrent operator лишь если он выигрывает **на устройстве**, особенно после 8K.

В репозитории GDN2 реализован как последовательный референс для проверки формулы, а
не как утверждение о скорости.

## 5. Где брать «ум», если параметров мало

Архитектура даёт проценты, но основная разница придёт из системы и обучения:

- **overtraining на качественных данных**: SmolLM2 использовал около 11T tokens для 1.7B;
- **pretraining distillation**: LFM2 хранит только teacher top-32 logits и раздельно учит
  вероятность попасть в top-K и распределение внутри него;
- **capacity alignment**: модели ≤3B часто хуже учатся на длинных traces сильного teacher,
  чем на коротких traces или умеренном teacher;
- **agent distillation**: 0.5B–3B модели выигрывают, когда учат не голый CoT, а корректные
  retrieval/code trajectories;
- **инструменты**: вычислять арифметику и получать свежие факты дешевле, чем запоминать;
- **test-time compute**: дополнительные latent loops или несколько проверяемых samples только
  для сложных запросов;
- **структурированная память**: пользовательские факты не должны растворяться в весах.

## 6. Текущий приоритет гипотез

### Tier A — немедленно

1. 350M conv/GQA dense baseline, 65K multilingual tokenizer, INT4 QAT.
2. 200M shared-depth вариант с 1–6 latent loops.
3. teacher Top-K pretraining distillation + обычный next-token loss.
4. retrieval/code/calculator trajectories и строгий tool grammar.
5. отдельная provenance-aware память пользователя.

### Tier B — после baseline

1. MobileMoE-подобные 60 fine-grained experts, top-4 + shared expert.
2. Engram: sweep размера table и точки injection.
3. GDN2 против conv на 8K/32K и реальном ARM.
4. MTP draft head и self-speculative decoding.

### Tier C — рискованные исследования

1. native ternary backbone;
2. byte-patch tokenizer;
3. mHC в sub-billion масштабе;
4. CSA/HCA-подобное sequence compression;
5. learned per-token recursion/early exit.

Ни одна Tier C идея не попадёт в крупный run без выигрыша на нескольких proxy scales.

# Программа экспериментов

## Принцип

Большой training run — последняя, а не первая стадия. Каждая новая идея проходит
matched ablation по параметрам, train FLOPs, данным и target-device wall-clock. Результат
без нескольких seeds и confidence interval считается наблюдением, а не выводом.

## Метрики Pareto

### Качество

- validation bits/byte и perplexity по доменам;
- instruction following (IFEval + собственные schema tests);
- русский/английский QA и перевод;
- dynamic arithmetic/logic, GSM-Plus/RUP-style perturbations;
- generated MQAR, stack/state tracking, RULER subsets;
- retrieval faithfulness, answerability и abstention;
- tool selection, valid arguments, multi-turn recovery;
- LongMemEval/LoCoMo-подобные задачи с contradiction/time ordering.

### Система

- cold/warm load time, RSS и model-mapped bytes;
- TTFT p50/p95;
- prefill tokens/s на 256/1K/4K/8K;
- decode tokens/s после 1K/4K/8K prefix;
- joule/token и battery % на фиксированный сценарий;
- температура и throughput после 1, 5 и 15 минут;
- размер KV/recurrent state и peak temporary buffers.

### Надёжность

- ECE/Brier для confidence;
- hallucination rate на unanswerable/freshness tests;
- perturbation consistency;
- tool/result agreement;
- memory precision, recall, stale-fact и contradiction rates;
- jailbreak/tool-injection и private-memory leakage.

## Фаза 0 — воспроизводимый стенд

- фиксированные seeds и manifests;
- train/eval split по документам, near-dedup и decontamination;
- единый tokenizer interface;
- profiler для PyTorch, llama.cpp и ExecuTorch;
- nightly smoke suite и weekly device suite.

**Gate:** один command воспроизводит loss curve и device report с commit hash.

## Фаза 1 — архитектурные proxy runs

Масштабы: 20M, 50M, 100M non-embedding parameters. Не меньше 3 seeds.

Ablations:

1. attention-only vs conv/GQA 10:6;
2. deep-thin vs wide-shallow;
3. GQA head ratios 2/4/8;
4. shared-depth R=1/2/3 с random unroll training;
5. GDN, KDA, GDN2 и conv в matched hybrid 3:1;
6. Engram table size, n-gram order и injection layer;
7. dense vs 8-expert и 60-fine-expert MoE;
8. MTP depth 0/1/2;
9. AdamW vs Muon при tuned learning rate.

Сначала generated tasks и held-out LM loss, затем перенос на downstream. Идея проходит,
если улучшает aggregate quality при ≤5% regression latency либо улучшает latency при
≤0.3 point aggregate regression.

## Фаза 2 — tokenizer и data mixture

Tokenizers: byte-BPE 32K/48K/65K, SentencePiece unigram и небольшой BLT proxy.
Языки минимум: русский, английский, украинский, немецкий; code/JSON отдельно.

Измеряем:

- bytes/token, tokens/word, continued words;
- LM-head FLOPs и embedding parameters;
- длину реальных chat/tool prompts;
- редкие слова, опечатки, Unicode и смешение раскладок;
- downstream при iso-FLOP, не iso-token.

Data mixture подбирается RegMix/data-mixing-law proxy runs, а не интуитивными процентами.
Отдельно оптимизируются general web, books/science, code, math, dialogue/tool, Russian и
другие языки.

## Фаза 3 — первая серьёзная модель

Кандидаты: Dense B0 и Shared-Depth R1. Начальная цель 100–300B качественных tokens,
а не сразу 10T. Checkpoints каждые фиксированные FLOPs.

Продолжение разрешается, если scaling curve остаётся предсказуемой и downstream
качество растёт; иначе бюджет переносится на data correction. Полный trillion-token run
делается только после стабильного extrapolation на трёх compute budgets.

## Фаза 4 — capability distillation

1. Off-policy teacher top-32 logits во время pretraining.
2. SFT curriculum: short direct → decomposition → tools → multi-turn recovery.
3. Correct teacher trajectories сортируются по student NLL; слишком сложные не доминируют.
4. On-policy rollout студента; teacher оценивает именно посещённые student states.
5. RLVR только там, где reward проверяется кодом/solver/test suite.

**Запрет:** бесконтрольное копирование длинного teacher CoT. Для ≤3B это может ухудшать
обучение из-за learnability gap и style drift.

## Фаза 5 — quantization

Сравниваются BF16, W8A8, W4A8, W4A4, Q4_K-like weight-only и native ternary branch.
INT4 QAT — часть training schedule, не косметическая конвертация финального checkpoint.
Router, norms, embeddings/LM head и FC2 input проверяются на sensitivity; допустим mixed
precision.

**Gate:** quality regression ≤1% relative по основным задачам, отсутствие catastrophic
regressions и фактическое ускорение на target runtime.

## Фаза 6 — телефон

Минимум два Android SoC и два поколения iPhone. Каждый прогон:

1. reboot/rest или контролируемая thermal state;
2. warm-up;
3. пять повторов, фиксированные prompt/output tokens;
4. CPU/GPU/NPU backends;
5. foreground contention scenario;
6. 15-минутный sustained test.

Выбирается не модель с максимальным benchmark, а Pareto-набор: `fast`, `balanced`,
`deep-think`. Пользовательский runtime может менять число recurrent loops и precision по
thermal/memory pressure.

## Критерии остановки проекта/ветки

Ветка закрывается, если после двух tuning rounds:

- не выигрывает у простого baseline при matched wall-clock;
- требует operator, который на реальном устройстве медленнее dense;
- улучшает static benchmark, но не dynamic/perturbed version;
- ломает calibration, tool schema или memory safety;
- преимущество исчезает на другом seed или model scale.

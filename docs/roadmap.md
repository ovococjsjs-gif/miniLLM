# Карта от исследовательского стенда к реально обученной miniLLM

Статус на 2026-08-20. Цель — как можно раньше получить осмысленный русско-английский
checkpoint, не перепутав дорогой training run с ещё одним слабым proxy.

## 1. Точка старта

### Уже работает

- типизированная гибридная архитектура: GQA, causal convolution, shared depth, GDN2,
  Engram, dense/MoE FFN и MTP;
- учёт stored/active parameters, KV/state, FLOP и active-byte energy proxy;
- single-device packed-token trainer с atomic resume-safe checkpoints;
- corpus records, license/quality/PII filters, deduplication, contamination и manifests;
- byte-BPE trainer и multilingual tokenizer economics;
- distillation losses и fake-quantization/QAT primitives;
- reference generation, exact GQA/conv/GDN2 cache и bilingual completion smoke suite;
- память, permissioned tools, retrieval trust boundary и bounded assistant loop;
- 61 unit/integration tests и воспроизводимые малые результаты.

### Что на самом деле доказано

- На 1.74M proxy attention-only лучше hybrid по короткому held-out LM loss примерно на
  0.14, а 2-attention/4-conv быстрее и намного лучше на generated MQAR.
- Layer ratio пока нельзя выбрать по одному aggregate score.
- Shared-depth имеет большую between-seed variance.
- Support-aware n-gram shelf даёт 2.52% coverage при 98.19% accuracy, но ещё не даёт
  end-to-end ускорения.
- На реальном 31M-token stream шестирукий 5M screen прошёл gate: attention loss 7.0017,
  edge 7.1244 (ratio 1.0175), edge быстрее в CPU training на 14.2%. Это 76.8K tokens/run,
  а не достаточное обучение или device benchmark.
- Полный 20M Attention-pass на P100 завершил 949 steps без FP16 skips: validation loss
  4.7329, 22.0K tokens/s, 1.013 GiB peak allocation. Генерация стала словоподобной, но
  fixed suite осталась 0/8 — pipeline масштабируется, полезная модель ещё не получена.
- На d=64/T=256 parameter-matched convolution быстрее CPU attention в 2.52× и имеет
  128× меньший fixed decode state; sequential GDN2 reference медленнее attention в 21.8×.
- Naive R=1/2/4 shared depth не даёт monotonic quality: step conditioning улучшает
  one-hop pointer retrieval, но 2–4 hop composition остаётся около chance. Нужен явный
  intermediate-state objective, а не простое повторение блока.
- Статический energy proxy полезен для постановки эксперимента, но не заменяет телефон.

### Чего пока нет

- полезных pretrained/SFT весов: 20M Attention checkpoint стабилен, но всё ещё даёт 0/8;
- production-legal русско-английского training corpus нужного масштаба;
- финального product tokenizer, повторно проверенного на расширенной смеси;
- систематического downstream evaluation шире диагностического smoke suite;
- distributed/sharded trainer для позднего большого run;
- экспорта в реальный mobile runtime и измерений на телефоне.

Иными словами: **исследовательский каркас уже серьёзный, но модель как продукт ещё не
обучена**. Текущие 350M/1.2B конфигурации — бюджеты и гипотезы, а не проверенные модели.

## 2. Реалистичный продуктовый ориентир

Целевой пакет 0.7–1.0 GB лучше строить вокруг одной модели, а не трёх наборов весов:

- `fast`: короткий ответ, greedy/низкая температура, ранний tool/retrieval routing;
- `balanced`: обычный ответ с памятью и проверкой;
- `deep`: больше system-level planning/verifier loops и tool budget.

Shared neural depth может позже усилить различие режимов, но первый продукт не должен от
неё зависеть. При Q4 модель порядка 0.8–1.3B параметров оставляет часть бюджета на scales,
KV/state, tokenizer, runtime и память. Dense 350M остаётся быстрым baseline и ступенью
масштабирования, а не автоматически финальным размером.

## 3. Что обязательно сделать до первого реального обучения

Всего три gate, а не завершение всей программы.

### Gate A — модель можно использовать после обучения

Нужны:

1. загрузка config + checkpoint + tokenizer;
2. deterministic greedy generation и seeded sampling;
3. stop tokens и ограничение длины;
4. проверка равенства full-forward и cached decode logits;
5. KV cache для GQA и rolling state для convolution;
6. CLI для completion и набора фиксированных prompts.

Сначала допустим медленный reference decode. Оптимизация runtime не должна задерживать
первую проверку качества, но cache correctness нужен до выводов о скорости.

### Gate B — данные имеют право стать весами

Нужна corpus v1 со следующими свойствами:

- коммерчески совместимые или явно разрешённые источники;
- отдельные RU/EN/domain proportions и документные IDs;
- train/validation/test split до packing;
- exact/near dedup и protected-eval decontamination;
- sharded streaming format вместо одного материализованного Python tuple;
- manifest с license, hashes, bytes и token counts;
- отдельная смесь web/encyclopedic, books/science, code, math, dialogue/tools.

Текущий UD corpus остаётся research proxy: он мал, смещён в русский и частично NC.

### Gate C — tokenizer и evaluation заморожены до запуска

- tokenizer 16K/32K/48K сравнивается на выборке именно corpus v1;
- основной кандидат сейчас — 32K, но решение принимается по RU/EN/code/JSON economics;
- special tokens и chat/tool serialization фиксируются до packing;
- до training сохраняются eval prompts и machine-readable gates;
- минимум: validation BPB/PPL по доменам, RU/EN completion, schema/tool tests,
  generated recall/state tracking и contamination report.

## 4. Ускоренная лестница обучения

Большой run не должен быть первой точкой, где обнаруживается ошибка данных или objective.
Но и оставаться на 300-step toy runs больше нельзя.

| Ступень | Модель | Порядок training tokens | Что решаем |
|---|---:|---:|---|
| L0 — текущая | 1.7M | 0.15M/run | plumbing и грубая отбраковка |
| L1 — первый real pilot | 10–30M | 0.1–0.6B | coherent generation, data/objective bugs |
| L2 — scaling checkpoint | 50–100M | 1–3B | architecture/data ranking и scaling curve |
| L3 — серьёзный baseline | 300–400M | 7–15B | базовая полезность, SFT/distillation readiness |
| L4 — product candidate | 0.8–1.3B | 30–100B сначала | баланс интеллекта, Q4 footprint и phone speed |

Это стартовые диапазоны, а не обещание качества. Продолжение каждого run разрешается по
loss/downstream scaling, а не потому, что заранее куплен весь token budget. Старый
ориентир 100–300B tokens относится к позднему overtraining product candidate, не к
первому серьёзному checkpoint.

## 5. Минимальный набор архитектурных рук

Чтобы быстрее перейти к данным, временно замораживаем широкий поиск.

### На L1 сравниваются только две руки

1. **Quality control:** обычная attention-heavy модель.
2. **Edge control:** matched hybrid attention/conv.

Одинаковы tokenizer, non-embedding parameters, tokens, optimizer и eval. Сначала короткий
multi-seed screen, затем длиннее обучается победитель и одна контрольная рука.

### Пока не допускаются к дорогому run

- GDN2 без chunkwise kernel;
- fine-grained MoE без grouped-GEMM/mobile prototype;
- shared-depth как основной путь до стабилизации нескольких seeds;
- predictive coding, sigma-delta training и gradient filtering;
- hard n-gram bypass.

Они не удаляются, а перестают блокировать baseline.

## 6. Что нужно усилить в trainer до L2

Для L1 уже готовы BF16/FP16 autocast, gradient checkpointing, fused AdamW, token-based
schedule, NaN/gradient guards, throughput/peak-VRAM logs и checkpoint v3 с
corpus/tokenizer/commit identity. До L2 остаются:

- sharded/streaming token loader и deterministic shard resume;
- update norm и validation-drift guards;
- фактический GPU memory/throughput benchmark и cost projection;
- DDP/FSDP только если выбранное железо действительно требует нескольких GPU.

Сложный distributed stack не должен задерживать L1 на одной GPU.

## 7. Evaluation, которая выбирает модель

Validation loss необходим, но недостаточен. Для каждой ступени сохраняется одна таблица:

- BPB/PPL: RU, EN, code/JSON, math/science, dialogue;
- generated recall, state tracking и perturbation consistency;
- короткие RU/EN completions с blind/manual rubric;
- tool name, JSON schema, arguments и grounded final answer;
- answerability/abstention и retrieval faithfulness;
- parameter count, train FLOPs, peak VRAM, tokens/s;
- после L3: Q8/Q4 regression, TTFT/decode, RSS и sustained phone run.

Выбор архитектуры делается по Pareto, а не по одному среднему score.

## 8. Последовательность ближайших работ

### Блок 1 — inference/eval vertical slice — выполнен

- model checkpoint loader и компактный inference checkpoint;
- greedy/seeded reference generation;
- exact cached attention/conv/GDN2 decode и equivalence tests;
- completion/evaluation CLI;
- фиксированный bilingual smoke set.

Engram пока использует корректный full-prefix fallback. Reference CPU benchmark получил
около 2.8× против полного пересчёта prefix на 1.74M proxy; mobile/runtime выводов из него нет.

**Результат:** любой будущий checkpoint можно сразу прочитать, сравнить и показать.

### Блок 2 — corpus v1 и tokenizer freeze — первый real pilot выполнен

Готовы machine-enforced source/license registry, conservative production policy,
disk-backed dedup, deterministic shards, protected-eval filtering, GitHub/Common Corpus
importers, tokenizer candidate/freeze scripts и hashed token packing.

Pinned GitHub pilot собран из OANC mirror, reviewed Russian public-domain literature и
RusDraCor: 8,560 документов / 142.7 MB, 47.6% EN и 52.4% RU по bytes. Выбран отдельный
8K L1-tokenizer; упаковано 31.09M train tokens с долями 59.3% EN / 40.7% RU. Tokenizer
сохранён в `artifacts/tokenizer-github-pilot-v1`, полный отчёт —
`results/github_pilot_data.json`.

Осталось до corpus v1, достаточного для длинного L1:

- увеличить train budget минимум до 0.1B без простого повторения эпох;
- добавить современный permissive Russian, science, code и dialogue/tools;
- расширить ручной accepted/rejected audit;
- повторить vocabulary decision после расширения смеси;
- закрепить окончательный attribution/release review.

**Текущий результат:** реальный переносимый pipeline/scaling checkpoint уже можно
обучать; полезную base model на 31M tokens обещать нельзя.

### Блок 3 — L1 training package — Attention GPU pass выполнен

Matched 4.86M/19.60M controls, mixed-precision path, checkpoint v3 и three-seed screen
готовы. Attention обработал 31.10M tokens на Tesla P100 за 23.55 минуты: loss 4.7329,
22.0K tokens/s, 1.013 GiB peak allocation, ноль skipped FP16 steps. Fixed completions
остались 0/8, но вместо newline collapse появились словоподобные RU/EN fragments.

Осталось:

- выполнить `kaggle_l1_edge_training.ipynb` на такой же P100;
- сравнить loss, throughput, peak VRAM и generation при matched recipe;
- расширить unique corpus до 0.1B+ без повторения той же узкой эпохи;
- после data expansion перейти к 50–100M scaling checkpoint.

**Следующий результат:** измеренная Attention/Edge GPU Pareto-точка.

### Блок 4 — L2 и выбор backbone

- масштаб 50–100M;
- data-mixture ablations малыми продолжениями;
- MTP 0/1 и Engram только как изолированные ablations;
- выбор backbone и data recipe для 300–400M.

### Блок 5 — assistant post-training

Только после появления адекватной base model:

- short capacity-aligned SFT;
- deterministic tool and memory trajectories;
- sparse teacher Top-K distillation;
- on-policy correction на состояниях студента;
- policy/control adapters отдельно от capability training.

### Блок 6 — quantization и телефон

- BF16/Q8/Q4 quality comparison;
- QAT sensitivity;
- export/runtime mapping;
- два Android SoC и два поколения iPhone;
- fast/balanced/deep system policies под latency/thermal budget.

## 9. Что не делать прямо сейчас

- не запускать 350M на текущем UD token stream;
- не выбирать финальную архитектуру по 1.7M/300-step результату;
- не строить новый экзотический optimizer до обычного real-data baseline;
- не смешивать pretraining, SFT, tools и policy в один неразбираемый run;
- не объявлять sparse/event FLOPs мобильным ускорением без kernel и watts;
- не ждать завершения mobile runtime, чтобы начать L1.

## 10. Ближайшее решение

Следующее критическое решение — **завершить matched 20M Edge-control на Tesla P100** без
изменения recipe после просмотра Attention. Затем loss/throughput/VRAM сравниваются одной
таблицей. Независимо от победителя corpus расширяется до 0.1B+ unique tokens: ещё одна
эпоха по тем же 31M узких tokens даст худшее evidence, чем новые данные. Attention пока
остаётся quality leader, но product backbone не выбирается до Edge GPU result.

# MiniLLM Lab

Исследовательский стенд для маленьких, быстрых и управляемых языковых моделей,
которые можно запускать локально, в том числе на телефонах.

Это не попытка «уменьшить Llama и надеяться». Текущая основная гипотеза —
**AIra-v2, событийно запускаемая когнитивная подложка**. Она разделяет задачу на
пять независимо измеряемых частей:

1. **L0-триггер** — компактная byte/character-полка без нейронной проверки выдаёт только статистически надёжные продолжения;
2. **эпизодическая память** — одношаговая запись, familiarity/margin rejection, provenance и удаление;
3. **остаточное нейронное ядро** — обрабатывает новизну, неоднозначность и композицию;
4. **эскалация** — неопределённость включает более глубокий проход, retrieval или детерминированный инструмент;
5. **локальная адаптация** — soft-residual обучение сейчас, PC-ALM/μPC только после прохождения matched controls.

Обычная маленькая decoder-модель сохранена как контрольный инструмент, а не как
главная исследовательская новизна.

## Что уже реализовано

- типизированные и проверяемые конфигурации архитектур;
- GQA + RoPE, мобильный gated short convolution и SwiGLU;
- sparse MoE с routed/shared experts (референсная реализация);
- hashed n-gram conditional memory по мотивам DeepSeek Engram;
- читаемая реализация рекуррентного Gated Delta Rule-2;
- shared-depth core с изменяемым числом итераций без роста числа весов;
- sequential Multi-Token Prediction и decoupled Top-K distillation losses;
- INT4/INT8 fake-quantization для QAT experiments;
- temporal SQLite memory и FTS document retrieval с provenance/privacy;
- permissioned agent loop, строгий JSON protocol, calculator/calendar tools;
- prompt-injection marking и проверка реальных citations;
- machine-enforced source/license policy, deterministic shards и SQLite-backed dedup;
- corpus manifests, quality/PII filters и protected-eval decontamination;
- multilingual byte-BPE candidates и corpus-bound tokenizer freeze;
- pinned GitHub RU/EN pilot: 142.7 MB, 31.1M train tokens, frozen 8K tokenizer;
- uint32 token packing и resume-safe checkpoint v3 с восстановлением всех RNG;
- BF16/FP16, gradient checkpointing, token-based schedule и non-finite guards;
- matched 5M/20M attention и edge L1 training controls;
- калькулятор параметров, INT4-памяти, KV-cache, recurrent state и FLOP/token;
- Fermi-разложение decode energy по активным весам, KV/state traffic и MAC;
- exact cached GQA/conv/GDN2 decode, seeded generation и checkpoint smoke suite;
- support/Wilson-gated компактная AIra byte/char shelf с frozen-holdout отчётами;
- доменная калибровка precision на отдельном split и нормализованная shelf/neural смесь;
- два decode-контроля: exact KV catch-up (не экономит весь active compute) и настоящий bounded byte-event core без скрытого state catch-up;
- deterministic byte↔ByteLevel-BPE bridge, включая dynamic BPE patches на произвольной byte boundary;
- bounded byte-event neural cores: matched gated-MLP, conv и attention patch controls, byte-output без обязательного прохода по shelf-позициям;
- lossless event packer для literal/shelf/source-copy spans и multi-byte head до 8 байтов;
- importance-sampled predictable control stream с unbiased full-loss estimator;
- hash-bound on-policy Top-K teacher records для точного student-generated prefix;
- AI Babysit critique/correction/verifier/preference records с teacher/student identity;
- project-owned `AIra Mentor v1`: 6K RU/EN verifier-first SFT records в 10 категориях;
- 1.7M-parameter/300-step local interaction smoke и первый 200-task Babysit failure set;
- Teacher Foundry: 11 причинных failure clusters, 11 RU/EN skill patches и 1,193 contrastive curriculum records;
- второй Qwen on-policy Foundry cycle: 17 exact corrections + 1,000 fresh seed-47 records без protected-split leakage;
- общий strict verifier с restricted-AST Python unit tests для training/Babysit/donor evaluation;
- recovered exact 532.5MB Unsloth Qwen3.5-0.8B Q4 donor/control, local llama.cpp runtime и OpenAI-compatible provider adapter;
- public-API real-state probe: exact 18-layer Gated DeltaNet/conv cache chains, 24 hidden outputs и full future logits;
- 48-record projected real-state dataset и 300-step patcher control с held-out MSE ratio 0.7421;
- masked recurrent-state patcher с future-KL/confidence objective и успешным 200-step synthetic dynamics proxy;
- deterministic event shards и resume-safe ≤300-step multi-head trainer;
- строгая UTF-8 grammar mask и отдельная calibration на generated contexts до разрешения bypass;
- request-level `AIraCascade`: accepted explicit fact возвращается напрямую, unknown/conflict падает в shelf→neural;
- bounded bipolar associative memory с explicit structured keys, unknown/conflict rejection, provenance, overwrite и удалением;
- soft-residual loss с ненулевым all-token control stream и matched 300-step сравнением;
- математический autograd-референс finite PC/PC-ALM и gradient-alignment benchmark;
- автономный stress test полки с oracle fallback, отдельно от teacher-forced coverage;
- generated и real-text proxy experiments;
- тесты causal-инвариантности, памяти, tools, distillation, QAT и учёта параметров.

Референсный PyTorch-код нужен для проверки идей. Он **не является мобильным
runtime**: для реального устройства понадобятся llama.cpp/ExecuTorch/MLX и fused
kernels.

## Быстрый старт

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'

# Сравнить четыре архитектурных бюджета на контексте 8K
minillm compare \
  configs/edge_dense_350m.json \
  configs/edge_recursive_200m.json \
  configs/edge_moe_1b3_a200m.json \
  configs/hybrid_gdn2_300m.json \
  --context 8192

# Подробный расчёт одной конфигурации
minillm analyze configs/edge_recursive_200m.json --context 8192 --recurrences 3

# Разложить Fermi energy proxy по active weights, KV/state traffic и MAC
minillm energy configs/edge_recursive_200m.json --context 8192 --recurrences 3

# Проверить полный forward/backward без загрузки датасета
minillm smoke-train configs/toy.json --steps 8

# Тесты
pytest

# Воспроизвести project-owned AIra Mentor v1 (6K verifier-first SFT)
python scripts/build_aira_mentor_dataset.py --overwrite

# Tiny interaction smoke и сбор свежих AI Babysit ошибок (оба capped at 300)
python scripts/train_aira_mentor_tiny.py --steps 300
python scripts/collect_aira_mentor_babysit.py

# Свернуть on-policy ошибки в teacher packet, skill patches и contrastive curriculum
python scripts/build_aira_teacher_foundry.py

# Matched negative control: ещё ≤300 steps tiny smoke на Foundry curriculum
python scripts/finetune_aira_mentor_foundry.py --steps 300

# Воспроизвести bounded proxy для event-state catch-up (не более 300 шагов)
python scripts/run_aira_state_patcher_proxy.py --steps 200

# Собрать pinned llama.cpp и взять hash-matched Unsloth Q4 из GitHub LFS mirror
.venv/bin/pip install -e '.[runtime]'
python scripts/bootstrap_qwen35_donor.py \
  --build-runtime --source github --download

# После запуска llama-server — bounded balanced donor baseline
python scripts/evaluate_qwen35_donor.py \
  --endpoint http://127.0.0.1:8080 --examples-per-category 5 --max-tokens 256

# Проверить найденный JSONL-датасет по AIra handoff contract
python scripts/audit_aira_dataset.py /path/to/candidate.jsonl

# Измерить event-packing upper bound до обучения
python scripts/benchmark_aira_event_packing.py \
  --train /path/to/train.txt --validation /path/to/validation.txt

# Подготовить hash-bound event shards и проверить trainer (локально <=300 шагов)
python scripts/prepare_aira_event_data.py \
  --shelf-text /path/to/shelf.txt --train-text /path/to/train.txt \
  --validation-text /path/to/validation.txt --tokenizer /path/to/tokenizer.json \
  --output /path/to/event-data
python scripts/train_aira_event_proxy.py --event-data /path/to/event-data --steps 300

# AIra-v2: собрать tokenizer-bound UTF-8 shelf для bridge-control decode
# (перед применением нужен отдельный frozen/domain calibration report)
python scripts/build_aira_shelf.py \
  --text /path/to/train.txt \
  --tokenizer artifacts/tokenizer-github-pilot-v1/tokenizer.json \
  --output artifacts/aira-shelf.npz

# Затем: minillm generate CHECKPOINT --tokenizer TOKENIZER --prompt TEXT \
#          --aira-shelf artifacts/aira-shelf.npz --aira-byte-bpe-bridge --json

# AIra-v2: воспроизвести PC-vs-PC-ALM alignment proxy
python scripts/benchmark_pc_alm.py

# AIra-v2: настоящий byte-event vertical slice (не более 300 шагов)
python scripts/run_aira_byte_event_proxy.py \
  --train-tokens /path/to/train.bin \
  --validation-tokens /path/to/validation.bin \
  --tokenizer artifacts/tokenizer-github-pilot-v1/tokenizer.json

# AIra-v2: matched MLP/conv/attention + curriculum ablation
python scripts/run_aira_event_core_ablation.py \
  --train-tokens /path/to/train.bin \
  --validation-tokens /path/to/validation.bin \
  --tokenizer artifacts/tokenizer-github-pilot-v1/tokenizer.json

# AIra-v2: full/hard/soft residual control (не более 300 шагов)
python scripts/run_aira_residual_proxy.py \
  --train /path/to/wikitext-2/train.txt \
  --validation /path/to/wikitext-2/valid.txt \
  --steps 300

# Исторический маленький MQAR-скрининг
PYTHONPATH=src python scripts/run_toy_mqar.py --steps 300 --seeds 123 456 789
```

## Текущая рабочая гипотеза

Цель — локальный RU/EN-ассистент примерно в **0.7–1 GB** с режимами
fast/balanced/deep, но преимущество должно появиться не от ещё одной обычной
маленькой LLM. AIra-v2 должна платить за нейронное вычисление пропорционально
новизне:

```text
raw byte/char shelf → episodic memory → residual neural core → deep/tool
```

На frozen WikiText-2 строгая UTF-8-полка уже даёт примерно 23% покрытия при 98.8%
teacher-forced accuracy для Wilson n2/p90; более строгий автономный Wilson p95 proxy
сохраняет 99.6% shelf precision при 15.3% фактических bypass-токенов. На русском
cross-book строгий автономный proxy даёт лишь 7.2% bypass при 98.2% precision — это
полезный, но небольшой выигрыш, а не основание для заявления об универсальном
ускорении. Доменная калибровка на отдельной размеченной половине повышает cross-book
UTF-8 coverage до 10.36% при 96.51% precision на финальной половине; она не заменяет
unlabeled OOD detector.

Matched 300-step контроль показал ожидаемый порядок: full loss лучше всего,
hard-filter хуже, soft-residual находится между ними. Значит, мягкий поток исправляет
голодание исходного AIra, но пока не даёт экономии обучения и не должен заменять full
loss. PC-ALM существенно лучше старого finite PC по совпадению с BP, однако простой
16-слойный tanh-референс требует около `8L` шагов для global cosine около 0.95; это
ещё не эффективный локальный kernel.

Первый byte/BPE vertical slice обнаружил важную архитектурную ошибку: byte shelf,
опрашиваемая только на BPE boundaries, сохраняет всего 1.28% coverage, хотя 96.88%
её токен-кандидатов имеют правильный byte-prefix. Исправленный event core запускается
на каждой неопределённой byte boundary, динамически BPE-сжимает только последние 64
байта и предсказывает следующий байт. На отдельном validation proxy он снизил proper
perplexity с 160.87 до 156.04, поднял accuracy с 21.95% до 23.12% и пропустил 2.61%
neural calls. Но unfused Python cascade медленнее, а при автономной генерации слабое
300-step ядро быстро уходит с manifold: shelf используется на 0.31% позиций с mean
precision лишь 38.78%. Отдельная generated-context calibration не находит ни одного
95%-safe threshold и правильно отключает bypass; oracle fallback сохраняет 95.32%
shelf precision при 2.67% coverage.

Matched 300-step ablation сравнил 471K gated-MLP, 465K conv и 475K attention, а также
random/contiguous/noise/recovery curricula. Настоящий 50/50 recovery phase повышает
autonomous accuracy до 7.80%, но ухудшает static cascade ppl до 164.34, accuracy до
19.76% и требует примерно в 18× больше training wall-time. Conv получает 7.71%
autonomous accuracy, но ppl 172.90. Ни один из 18 runs не получил safe generated-context
threshold, поэтому gated-MLP остаётся быстрым baseline, а ни один новый core не принят.
Расширение unique sampling window с 0.8M до 8M BPE tokens при
том же числе examples/steps улучшает static cascade ppl до 149.11 и accuracy до 23.88%,
но autonomous accuracy остаётся 7.38% и safe threshold всё ещё отсутствует. Более
разнообразные данные приняты как новый training default, но сами по себе exposure drift
не исправляют.

Offline event packer показывает до 8.00× сокращения только как oracle upper bound для
8-byte head. Текущая строгая shelf копирует лишь 3.17% байтов и сама по себе даёт около
1.03× upper bound neural-call reduction; короткие shelf spans даже дробят 8-byte events.
Prompt-copy покрывает 42% при min-2, но ухудшает event count; даже copy spans от 16
байтов не превосходят pure oracle 8-byte event count при включённой короткой shelf. Значит, главный следующий gate —
не таблица и не copy сами по себе, а реально обученный и откалиброванный multi-byte head.

Новая pretrained ветка не назначает маленькую open model учителем. Arena.ai agent формирует причинные `SkillPatch`, а solvers/verifiers размножают их в свежие задачи. Первый matched Foundry intervention снизил tiny validation ppl с 2.427 до 2.222, но на одинаковых свежих seed-45 задачах strict pass остался 0/10 → 0/10: curriculum исправляет supervision, но не заменяет pretrained capacity. Exact Unsloth Qwen3.5-0.8B Q4 восстановлен из 50 частей и принят только как 532.5MB language donor/control. На двух CPU threads он даёт около 21.4 generated token/s при ~852 MiB peak RSS. Balanced protected sample получает 6/50 strict, 31/50 content и 0/18 обязательных source attributions; fresh seed-46 rollout — 3/20 strict. Answer-free protocol control поднял source 0/7 → 3/7, content 12/20 → 13/20 и protocol 17/20 → 18/20, но strict остался 3/20, поэтому prompt-only fix отклонён. Это подтверждает, что donor даёт язык, но не заменяет teacher/verifier. Его шесть групп `3×Gated DeltaNet + attention` остаются конкретной точкой для fast/balanced/deep exits и state catch-up. Public llama.cpp callback probe теперь извлекает реальные `128×128×16` states всех 18 recurrent layers, conv caches, 24 hidden outputs и полные 248,320-way logits. Между prompt и первым autoregressive event получено 18/18 byte-exact state transitions и 18/18 conv transitions; средний относительный state delta равен 0.2056. Следующий 48-record real-state control подтвердил 864/864 cache links; 69.6K patcher снизил held-out projected-state MSE с 0.6193 до 0.4596 (ratio 0.7421), обойдя copy и mean-delta. Это CountSketch learnability evidence, а не injectible full-state reconstruction, поэтому acceleration claim закрыт.

Плотные 350M, recurrent 209M, MoE и GDN2-конфигурации остаются контрольными ветками.
Их обычное масштабирование приостановлено, пока end-to-end каскад не покажет лучший
quality-adjusted active compute.

## Документы

- [Ускоренная карта до реально обученной модели](docs/roadmap.md)
- [Reference generation, cache и checkpoint evaluation](docs/inference.md)
- [Corpus v1: источники, policy, shards и tokenizer freeze](docs/data-v1.md)
- [L1: matched screen и первый 20M training package](docs/l1-training.md)
- [Запуск полного L1 на Kaggle GPU](docs/kaggle-l1.md) — notebook и data bundle в [`kaggle/`](kaggle/)
- [Малые тесты mixer scaling и adaptive depth](docs/small-scale-experiments.md)
- [Карта исследований и проверенные факты](docs/research-landscape.md)
- [Предлагаемая архитектура и система памяти](docs/architecture.md)
- [План экспериментов и критерии остановки](docs/experiments.md)
- [Данные, distillation и post-training](docs/training.md)
- [Результат первого MQAR-скрининга](docs/toy-mqar-result.md)
- [Канонический аудит и план AIra-v2](docs/aira-v2-audit.md)
- [Контракт данных и готовность AIra base training](docs/aira-training-readiness.md)
- [Аудит synthetic SFT-кандидатов и AI Babysit](docs/aira-synthetic-sft-audit.md)
- [AIra Mentor v1: 6K verifier-first RU/EN SFT](docs/aira-mentor-v1.md)
- [AIra Teacher Foundry v1: failure clusters → skill patches → curriculum](docs/aira-teacher-foundry-v1.md)
- [Qwen3.5-0.8B как language donor/control и state-catch-up gate](docs/aira-qwen35-donor.md)
- [реальный Qwen3.5 recurrent-state/full-logit probe](docs/aira-qwen35-real-state-probe.md)
- [Исторический разбор AIra (superseded)](docs/aira-review.md)
- [Аннотированные первоисточники](docs/sources.md)

> Название репозитория задано заранее. Проект не связан с одноимённой работой
> **MiniLLM: On-Policy Distillation of Large Language Models** (Gu et al.).

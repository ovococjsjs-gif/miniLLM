# MiniLLM Lab

Исследовательский стенд для маленьких, быстрых и управляемых языковых моделей,
которые можно запускать локально, в том числе на телефонах.

Это не попытка «уменьшить Llama и надеяться». Проект разделяет задачу на пять
независимо измеряемых частей:

1. **вычислительное ядро** — гибрид дешёвых causal-conv и редких GQA-слоёв;
2. **адаптивное мышление** — повторно используемое по глубине ядро;
3. **знания и память** — learned Engram-память отдельно от изменяемой памяти пользователя;
4. **точность действий** — retrieval, калькулятор, код и другие проверяемые инструменты;
5. **обучение** — качественные данные, capacity-aligned distillation, MTP и INT4 QAT.

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
- uint32 token packing и resume-safe proxy trainer с восстановлением всех RNG;
- калькулятор параметров, INT4-памяти, KV-cache, recurrent state и FLOP/token;
- Fermi-разложение decode energy по активным весам, KV/state traffic и MAC;
- exact cached GQA/conv/GDN2 decode, seeded generation и checkpoint smoke suite;
- support/confidence-gated n-gram draft shelf без неявной замены neural policy;
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

# Воспроизвести маленький MQAR-скрининг
PYTHONPATH=src python scripts/run_toy_mqar.py --steps 400 --seeds 123 456 789
```

## Текущая рабочая гипотеза

Первый серьёзный baseline — **плотная модель около 350M**: 10 дешёвых conv-блоков,
6 GQA-блоков, tied 65K embeddings, MTP и последующий INT4 QAT. Расчёт для текущей
конфигурации: 346M хранимых параметров, теоретически 165 MiB чистых INT4-весов и
48 MiB KV-cache при 8K (без runtime overhead).

Главный исследовательский вариант — **209M хранимых параметров с рекуррентной
глубиной**. При трёх проходах он применяет около 301M активных параметров на токен,
но занимает теоретически около 100 MiB в INT4. Число проходов можно выбирать по
сложности запроса.

MoE и GDN2/KDA остаются важными ветками, но не назначены baseline до измерения на
реальном телефоне: их теоретические FLOP-преимущества легко съедаются dispatch и
неоптимизированными kernels.

## Документы

- [Ускоренная карта до реально обученной модели](docs/roadmap.md)
- [Reference generation, cache и checkpoint evaluation](docs/inference.md)
- [Corpus v1: источники, policy, shards и tokenizer freeze](docs/data-v1.md)
- [Карта исследований и проверенные факты](docs/research-landscape.md)
- [Предлагаемая архитектура и система памяти](docs/architecture.md)
- [План экспериментов и критерии остановки](docs/experiments.md)
- [Данные, distillation и post-training](docs/training.md)
- [Результат первого MQAR-скрининга](docs/toy-mqar-result.md)
- [Разбор AIra и безопасно принятые идеи](docs/aira-review.md)
- [Аннотированные первоисточники](docs/sources.md)

> Название репозитория задано заранее. Проект не связан с одноимённой работой
> **MiniLLM: On-Policy Distillation of Large Language Models** (Gu et al.).

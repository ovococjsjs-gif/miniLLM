# miniLLM — AIra-Qwen

Цель проекта — получить маленькую локальную модель, которая одновременно:

- заметно быстрее и дешевле обычного dense decode;
- сохраняет или повышает качество Qwen3.5-0.8B;
- реально обучается на AI Babysit corrections изменением параметров;
- помещается примерно в 0.7–1 ГБ и в перспективе работает на телефоне.

## Что считается AIra

Активная гипотеза — адаптировать **сам neural computation graph Qwen**:

1. learned event/span policy определяет, где полный token-by-token compute избыточен;
2. learned full-state updater продвигает recurrent state через пропущенный участок;
3. attention anchors и fallback сохраняют качество на сложных случаях;
4. AI Babysit собирает собственные ошибки изменённой модели;
5. correction SFT + preference/constitutional training обновляют параметры модели;
6. модель повторно проверяется на новых задачах, forgetting controls, скорости и памяти.

Цикл не засчитывается, если улучшение объясняется готовым ответом, keyword route или lookup table.

## Текущий честный статус

- Проверенный donor: Qwen3.5-0.8B Q4_K_M, 532.5 МБ.
- Native runtime: pinned llama.cpp на CPU.
- Получены реальные full recurrent states 18 Gated DeltaNet layers, conv caches, hidden outputs и full logits.
- Public partial-state API умеет возвращать изменённый recurrent state в Qwen.
- Stale-state и oracle-interpolation controls показывают, что updater должен быть очень точным.
- Projected state patcher доказал только learnability; он не разрешает ускорение.
- **Injectible learned full-state updater ещё не обучен.**
- **Recurrent skipping и production neural acceleration выключены.**

AIra One сохранён как controller для tools, memory, documents и запуска donor. Из него удалены stored-answer Babysit routes. Feedback записывается только как будущий training material.

## Что было удалено

Из активного дерева и package manifest удалены:

- `SkillShelf` и встроенные ответы по ключевым словам;
- broad `24/24` answer-cache experiment;
- ограниченный candidate-token output adapter;
- smoke-отчёты, где ускорение создавалось обходом Qwen через готовые ответы.

История остаётся в Git, но эти результаты больше не представляются как обучение AIra.

## Ближайший gate

Первый реальный архитектурный milestone:

```text
full Qwen state_before + emitted event
                 ↓
       learned low-rank updater
                 ↓
 predicted full state_after
                 ↓ public llama.cpp state injection
       future logits / generation
```

Только если learned state проходит held-out reconstruction, injected-logit и generated-quality controls, будет разрешён benchmark фактического пропуска recurrent compute.

## Быстрый старт

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'

# Восстановить проверенный donor и собрать runtime
python scripts/bootstrap_qwen35_donor.py --build-runtime --source github

# Локальный controller/chat
python scripts/run_aira_one.py

# Тесты
pytest
```

## Основные технические документы

- [Qwen donor и архитектурный план](docs/aira-qwen35-donor.md)
- [Реальные recurrent states](docs/aira-qwen35-real-state-probe.md)
- [AIra One как controller](docs/aira-one-v01.md)
- [Журнал решений](docs/decisions.md)

Старые proxy experiments остаются воспроизводимыми исследовательскими контролями, но не входят в текущий definition of done для AIra-Qwen.

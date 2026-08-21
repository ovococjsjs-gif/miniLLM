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
- Projected state patcher доказал только learnability и заменён full-state направлением.
- Первый 5.65M-parameter Gated Delta updater обучен на реальных состояниях за 300 шагов.
- Held-out full-state MSE снизился до `0.8428×` copy baseline.
- Stage-1 state-only injection снижала KL, но на 16 переходах state-only mean gate не удержался (`1.0277×` copy).
- Новый 2.42M-parameter convolution-row updater получил held-out MSE ratio `0.5581`.
- Без oracle convolution строгий 16-transition full-cache replay снизил mean KL `0.021935 → 0.015856` (`0.7229×`) и сохранил oracle argmax 16/16.
- KL улучшился только на 10/16 переходах; free-generation и speed gates не пройдены.
- **Recurrent skipping и production neural acceleration остаются выключены.**

AIra One сохранён как controller для tools, memory, documents и запуска donor. Из него удалены stored-answer Babysit routes. Feedback записывается только как будущий training material.

## Что было удалено

Из активного дерева и package manifest удалены:

- `SkillShelf` и встроенные ответы по ключевым словам;
- broad `24/24` answer-cache experiment;
- ограниченный candidate-token output adapter;
- smoke-отчёты, где ускорение создавалось обходом Qwen через готовые ответы.

История остаётся в Git, но эти результаты больше не представляются как обучение AIra.

## Ближайший gate

Текущий архитектурный путь:

```text
attention-anchor hidden + token event
        ↓                         ↓
key/value/gate/beta predictor   conv newest-row predictor
        ↓ exact state algebra     ↓ exact two-row shift
        └──────── full recurrent + conv cache ────────┘
                          ↓ public state injection
                       future logits
```

Full recurrent state и convolution cache теперь предсказываются без oracle для пяти слоёв; mean injected-logit gate пройден и в строгом full-cache control. Запас всё ещё мал, один validation prompt регрессирует, а вычисления слоя пока только заменяются post-hoc. Следующий gate — свободная многотокенная генерация и native benchmark физического пропуска compute. Speed claim разрешается только после generated-quality non-regression.

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

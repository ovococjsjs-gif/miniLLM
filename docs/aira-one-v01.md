# AIra One v0.1: локальный controller, не обученная AIra-модель

AIra One остаётся удобной оболочкой для запуска проверенного Qwen3.5-0.8B, инструментов, памяти и сбора feedback. После пересмотра проекта он **не считается** реализацией обучающейся AIra.

## Что оставлено

- интерактивный RU/EN CLI;
- OpenAI-compatible локальный API;
- точные calculator/algebra/logic/code/calendar tools;
- подтверждаемая SQLite-память с provenance и удалением;
- документы с границей между данными и инструкциями;
- fast/balanced/deep budgets;
- append-only журнал попыток и пользовательских исправлений;
- fallback в настоящий локальный Qwen для неизвестных запросов.

## Что удалено

Из runtime удалены `SkillShelf` и два встроенных Babysit answer routes. Исправление в журнале больше не превращается в готовый ответ по ключевым словам. Оно остаётся training record до тех пор, пока реальный optimizer не изменит параметры нейронной модели.

Точные tools не являются попыткой имитировать интеллект: калькулятор, память и вызов инструмента — нормальные компоненты ассистента. Но их результаты нельзя смешивать с neural-quality benchmark.

## Честная область старых измерений

Protected `174/174` и fresh `100/100` проверяют только поддерживаемые deterministic Mentor-family contracts. Они доказывают корректность controller, но ничего не говорят об общем качестве Qwen или AIra.

Результаты answer-cache экспериментов удалены из активного дерева. При необходимости они доступны в Git-истории, но больше не входят в package manifest и README.

## Запуск

```bash
python scripts/bootstrap_qwen35_donor.py --build-runtime --source github
python scripts/run_aira_one.py
```

Только deterministic tools без Qwen:

```bash
python scripts/run_aira_one.py --offline --prompt "Вычисли: 12 * (7 + 3)"
```

API:

```bash
python scripts/serve_aira_one.py --host 0.0.0.0 --port 8000
```

## Активная работа

Главная ветка исследований теперь строит **AIra-Qwen**, где обучаемые ускоряющие модули входят в neural computation graph Qwen. Первый full-state predictor уже обучен и через публичный llama.cpp API снизил средний held-out future KL относительно matched copy, но один prompt регрессировал. До реального recurrent skip ещё обязательны convolution, free-generation и speed gates. Только после архитектурной стабилизации начинается настоящий Babysit SFT + preference/constitutional fine-tuning.

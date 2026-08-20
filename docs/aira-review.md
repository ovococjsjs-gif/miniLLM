# Разбор ветки AIra и безопасные заимствования идей

Дата разбора: 2026-08-20. Рассмотрена ветка
[`arena/019fcef3-aira`](https://github.com/ovososjdjd-boop/AIra/tree/arena/019fcef3-aira),
локальный снимок `1bc5975`.

## Ограничение происхождения

В рассмотренной ветке не найден файл LICENSE/COPYING. Поэтому исходный код и данные
AIra не переносились. Ниже независимо реализованы только общие инженерные идеи, с
другими API, тестами и более консервативными границами применимости. Числа AIra не
считаются доказательством для miniLLM: значительная их часть получена на символьных или
синтетических потоках.

## Что принято

### 1. N-gram shelf только как источник draft-токенов

`src/minillm/draft.py` содержит opt-in shelf с двумя обязательными гейтами:

- минимальное число наблюдений контекста;
- нижняя граница Wilson для вероятности top-1 продолжения, а не одна сырая частота.

Кандидат по умолчанию должен совпасть с argmax основной модели
(`verify_greedy_candidate`). Следовательно, модуль сам по себе не меняет greedy policy.
Он не подключён к runtime автоматически и пока не обещает ускорения: для него нужен
реальный multi-token speculative decoder с пакетной проверкой.

До запуска был сохранён контракт
`configs/experiments/ngram_draft_proxy.json`. Полный train/validation поток 4K byte-BPE
дал для первичного гейта (Wilson 95%, lower bound ≥0.90):

- coverage **2.523%**;
- held-out accuracy **98.188%**;
- заранее заданный гейт coverage ≥0.5% и accuracy ≥95% пройден.

Более агрессивная сырая частота ≥0.90 при support ≥4 дала 6.683% coverage, но только
93.784% accuracy. Это важный отрицательный результат: один confidence threshold без
учёта статистической опоры недостаточно надёжен. Полная матрица находится в
`results/ngram_draft_proxy.json`, воспроизведение — `scripts/evaluate_ngram_draft.py`.

### 2. Энергия декодирования через активные байты

`src/minillm/energy.py` дополняет FLOP/token явным Fermi-разложением:

- чтение активных квантованных весов;
- чтение полного KV history;
- read+write рекуррентного состояния;
- арифметика MAC.

При placeholder-профиле LPDDR 60 pJ/byte и MAC 0.5 pJ статический расчёт на контексте
8K даёт:

| конфигурация | active weights | KV | state R+W | total proxy |
|---|---:|---:|---:|---:|
| dense 350M | 164.0 MiB | 48.0 MiB | 0 | 13.562 mJ/token |
| recursive 200M | 175.4 MiB | 40.0 MiB | 0 | 13.776 mJ/token |
| MoE 1.3B / active 200M | 108.0 MiB | 18.0 MiB | 0 | 8.079 mJ/token |
| hybrid GDN2 300M | 139.9 MiB | 24.0 MiB | 6.75 MiB | 10.924 mJ/token |

Это не измерение мощности. Расчёт намеренно штрафует каждый активный вес как чтение из
выбранного яруса памяти и не знает о cache residency, activation traffic, fused kernels,
dispatch, thermals и sampling. Поэтому преимущество MoE — только гипотеза для device
benchmark, а не решение о baseline. Отчёт: `results/decode_energy_proxy.json`; CLI:
`minillm energy CONFIG --context 8192`.

### 3. Resume-safe checkpoint v2

`train_proxy(..., resume_from=...)` теперь восстанавливает:

- модель и optimizer;
- Python, NumPy, torch CPU и CUDA RNG;
- состояние отдельного генератора выборки batches;
- step, best/last validation и расписание через строгую проверку `TrainConfig`;
- сигнатуры train/validation token files.

Конфигурации и данные проверяются до продолжения. Старый/неполный checkpoint не
принимается как будто он воспроизводим. Regression test сравнивает непрерывный 4-step
run и продолжение с step 2 побитно по всем model tensors. Обычный путь без `resume_from`
не изменён.

### 4. Машиночитаемые ставки

Первый контракт хранит гипотезу, primary gate, death criteria и non-claims отдельно от
результата. Это уменьшает возможность подобрать критерий после просмотра чисел. Такой
формат следует распространить на дорогие ablations, но он не заменяет несколько seeds,
confidence intervals и target-device замеры.

## Что отложено

| Идея AIra | Решение | Причина |
|---|---|---|
| predictive-coding / forward-only training | исследовательская ветка | пока есть quality gap, чувствительность к scale и дорогая relaxation dynamics |
| sigma-delta activation/event bus | simulator only | skipped operations в dense NumPy не равны ускорению или joule savings без sparse kernels |
| HDC как основная память | не заменять SQLite/FTS | нет provenance, temporal contradiction handling и privacy boundary уровня текущей памяти |
| gradient filtering shelf-предсказуемых токенов | не применять | может лишать neural model данных и создавать слабую «голодную» зону |
| hard trigger bypass | отклонено | heuristic probability меняет распределение основной модели и может ломать calibration |

## Решение

AIra полезна как источник проверяемых гипотез и дисциплины экспериментов, но не как
готовая архитектура. В production path приняты только изолированные, opt-in и
аудируемые механизмы. Baseline miniLLM не меняется: attention-only остаётся quality
control, а 2-attention/4-conv — edge control до реальных измерений на телефоне.

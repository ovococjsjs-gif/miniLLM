# Первый proxy experiment: generated MQAR

Команда:

```bash
PYTHONPATH=src python scripts/run_toy_mqar.py \
  --steps 400 --batch-size 32 --seeds 123 456 789
```

Задача на лету создаёт 4 пары key→value и задаёт 2 query. Test instances используют
отдельный RNG stream; шанс угадать token — 3.125%. Все варианты видят 12,800 training
examples на seed. Это **iso-step**, не iso-FLOP эксперимент.

| Variant | Stored params | Effective depth | Held-out token accuracy, mean | Between-seed std | CPU time/run |
|---|---:|---:|---:|---:|---:|
| 4 attention | 152,512 | 4 | 28.7% | 1.3% | 11.9 s |
| 3 conv + 1 attention | 153,184 | 4 | **97.4%** | 2.2% | **10.1 s** |
| 3 GDN2 + 1 attention | 214,288 | 4 | 29.9% | 1.5% | 33.9 s |
| recurrent conv+attention | 124,032 | 7 | 53.3% | 38.8% | 17.7 s |
| recurrent + Engram | 130,448 | 7 | 53.7% | 20.9% | 19.4 s |

Полные данные: [`results/toy_mqar.json`](../results/toy_mqar.json).

## Что можно заключить

- Реализация учится и generated benchmark различает архитектуры.
- На этой конкретной локально-структурированной задаче простой conv/GQA hybrid резко
  sample-efficient и быстрее остальных reference variants.
- Shared-depth варианты нестабильны между seeds. До scale-up им нужны отдельные
  исследования initialization, normalization и random-depth curriculum.
- Последовательный GDN2 reference нельзя использовать для вывода о production kernel speed.

## Чего заключить нельзя

- что convolution вообще умнее attention;
- что GDN2/KDA не работают на языке или длинном контексте;
- что Engram не помогает (таблица крошечная, задача и training budget ограничены);
- что ranking сохранится при iso-FLOP, больших моделях и реальном pretraining.

Практическое решение после эксперимента: conv/GQA остаётся B0 baseline; recurrence, Engram
и GDN2 должны выиграть отдельные matched tests, прежде чем усложнять большой run.

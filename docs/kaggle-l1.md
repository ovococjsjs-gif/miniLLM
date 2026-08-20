# Запуск L1 на Kaggle

Готовый notebook: [`notebooks/kaggle_l1_training.ipynb`](../notebooks/kaggle_l1_training.ipynb).
Он предназначен для Kaggle GPU Notebook и по умолчанию обучает 19.60M attention-control
на полном 31.09M-token pilot stream.

## Быстрый запуск

1. Импортировать `.ipynb` в Kaggle.
2. Открыть **Settings → Accelerator → GPU**.
3. Включить **Internet** для клонирования репозитория и pinned corpus sources.
4. Оставить `VARIANTS = ["attention"]` для первого запуска.
5. Выполнить **Run All**.
6. После завершения нажать **Save Version**: только так содержимое `/kaggle/working`
   станет постоянным output этой версии.

Notebook автоматически:

- клонирует закреплённую ревизию miniLLM;
- сравнивает compute capability GPU со списком kernels в установленном PyTorch;
- для Pascal P100 (`sm_60`) заменяет несовместимый Kaggle wheel на официальный
  `torch==2.7.1` с CUDA 12.6;
- выбирает BF16 только на Ampere+, а на T4/P100 — FP16;
- до обучения выполняет настоящую CUDA-операцию и fused AdamW smoke-test;
- воспроизводит corpus из трёх GitHub-источников на exact SHA;
- полностью проверяет SHA-256 всех token streams;
- делает dry-run с ожидаемыми 949 optimizer steps;
- обучает модель и сохраняет resume checkpoints;
- строит loss-графики и запускает фиксированную RU/EN completion suite;
- создаёт компактный inference export.

## Как не перестраивать данные

Предпочтительный путь — готовый архив `l1-github-pilot-data-v1.tar.gz` размером около
48 MB. Его нужно загрузить как Kaggle Dataset и подключить через **Add Input**. Notebook
находит архив автоматически, распаковывает и полностью проверяет SHA-256 каждого stream;
`DATA_INPUT` обычно менять не требуется.

Без bundle первое воспроизведение клонирует три GitHub repositories и временно занимает
примерно 0.5 GB сверх packed streams. Эта стадия зависит от Kaggle Internet и долго не
печатает прогресс во время `git fetch`. После проверки source checkouts, imported text и
corpus shards удаляются; остаётся каталог `/kaggle/working/minillm-l1-data/tokens`
размером около 132 MB. Его также можно сохранить как отдельный Kaggle Dataset и в следующем
notebook указать каталог в `DATA_INPUT`.

## Attention и edge

Первым запускается:

```python
VARIANTS = ["attention"]
```

Для последовательного matched-сравнения:

```python
VARIANTS = ["attention", "edge"]
```

Две руки нельзя обучать одновременно на одном GPU: это портит throughput/memory
измерения и увеличивает вероятность OOM.

## Resume между сессиями

Полные каталоги находятся в `/kaggle/working/minillm-runs`. Они содержат `best.pt` и
`step-*.pt` с optimizer/RNG/scaler state. После **Save Version** output старой версии можно
подключить через **Add Input** и указать read-only каталог:

```python
RESUME_RUNS = {
    "attention": "/kaggle/input/.../minillm-runs/attention",
    "edge": "",
}
```

Notebook копирует его в writable storage и продолжает с checkpoint максимального шага.
При resume нельзя менять precision, batch size, sequence length, accumulation, seed или
training schedule: checkpoint v3 намеренно отвергает такое смешение экспериментов.

Если CUDA завершилась до первого шага (например, старый Kaggle PyTorch не содержал
`sm_60`), остаётся пустой `metrics.jsonl` без checkpoint. Notebook распознаёт этот случай,
удаляет только пустой run и начинает заново после успешного CUDA smoke-test.

## Что сохранить

- `minillm-runs/<variant>/step-*.pt` — точное продолжение обучения;
- `minillm-runs/<variant>/best.pt` — лучший полный checkpoint;
- `minillm-export/<variant>/best-inference.pt` — компактная генерация;
- `metrics.jsonl`, `l1-summary.json`, `completion-smoke.json` — анализ;
- `kaggle-run-manifest.json` — commit, GPU, data hashes и полный план.

31M токенов — только scaling checkpoint. Даже успешный run не подтверждает качество
целевого ассистента 0.7–1 GB и не заменяет будущие 0.1B+, 1–3B и product-scale этапы.

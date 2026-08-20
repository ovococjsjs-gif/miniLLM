# Kaggle L1

В этой папке находятся оба файла для запуска:

- [`kaggle_l1_training.ipynb`](kaggle_l1_training.ipynb) — notebook;
- [`l1-github-pilot-data-v1.tar.gz`](l1-github-pilot-data-v1.tar.gz) — готовые packed
  RU/EN tokens, tokenizer и provenance manifests.

Для обычного запуска достаточно скачать/import только notebook и включить Kaggle
**Internet + GPU**. Notebook клонирует закреплённый commit этого репозитория, поэтому архив
окажется рядом с кодом автоматически. На стадии 3 он распакуется без клонирования трёх
исходных corpus repositories.

Data bundle:

- размер архива: около 48 MB;
- train: 31,094,503 tokens;
- validation: 1,178,019 tokens;
- test: 2,213,356 tokens;
- SHA-256: `5ccfc6aaf8b5cf1c4a6201a5dc0a92fdc24d16c80efcfddbd1ea3ac106412889`.

Notebook заново считает SHA-256 распакованных `.bin` перед обучением. Полная инструкция и
resume-процедура: [`docs/kaggle-l1.md`](../docs/kaggle-l1.md).

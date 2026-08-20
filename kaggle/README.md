# Kaggle L1

В этой папке находятся готовые файлы для запуска:

- [`kaggle_l1_training.ipynb`](kaggle_l1_training.ipynb) — первый Attention-run;
- [`kaggle_l1_edge_training.ipynb`](kaggle_l1_edge_training.ipynb) — следующий matched
  Edge-контроль на той же Tesla P100;
- [`l1-github-pilot-data-v1.tar.gz`](l1-github-pilot-data-v1.tar.gz) — готовые packed
  RU/EN tokens, tokenizer и provenance manifests.

Attention-run уже выполнен. Для следующего эксперимента нужно импортировать Edge notebook
и включить Kaggle **Internet + GPU**. Notebook требует P100 для честного сравнения,
клонирует закреплённый commit и распаковывает архив на стадии 3 без скачивания исходных
corpus repositories.

Data bundle:

- размер архива: около 48 MB;
- train: 31,094,503 tokens;
- validation: 1,178,019 tokens;
- test: 2,213,356 tokens;
- SHA-256: `5ccfc6aaf8b5cf1c4a6201a5dc0a92fdc24d16c80efcfddbd1ea3ac106412889`.

Notebook заново считает SHA-256 распакованных `.bin` перед обучением. Полная инструкция и
resume-процедура: [`docs/kaggle-l1.md`](../docs/kaggle-l1.md).

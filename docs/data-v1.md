# Corpus v1: permissive-first data path

Статус: pipeline реализован и первый pinned GitHub RU/EN pilot собран. Он готов для
короткого L1 scaling checkpoint, но слишком мал и доменно узок для полезной base model.
Этот документ — инженерная политика происхождения, а не юридическое заключение.

## 1. Главный принцип

Размер корпуса не даёт права потерять происхождение отдельных документов. Каждый
документ до token packing должен иметь:

- стабильный ID и source registry entry;
- каноническую лицензию;
- URL происхождения, когда нужны attribution или дополнительная проверка;
- язык, домен, дату получения и content SHA-256;
- документный split group, чтобы chunks одного произведения не утекали между splits.

`configs/corpus/source_registry.json` присваивает источнику один из статусов:

- `approved` — допускается production policy при разрешённой per-document license;
- `conditional` — нужен отдельный аудит/решение до попадания в releasable weights;
- `research-only` — допускается только явно выбранной research policy.

`configs/corpus/policy_production.json` принимает Public Domain, CC0, CC BY и небольшой
набор permissive software licenses. Она намеренно отклоняет NC, ND, ShareAlike, unknown и
расплывчатую метку `open license`.

## 2. Решения по источникам

### Основной кандидат: Common Corpus, только permissive subset

Common Corpus заявляет document-level license/provenance, RU/EN и несколько доменов:
<https://huggingface.co/datasets/PleIAs/common_corpus>. Импорт не доверяет общей карточке
как единственной лицензии: каждая строка повторно проходит локальный allowlist. Вход без
понятной canonical license или provenance URL отклоняется.

Разрешены только Public Domain, CC0, CC BY 4.0, MIT, Apache-2.0, BSD, ISC, 0BSD и
Unlicense. CC BY-SA из того же корпуса не проходит через production entry.

### Approved narrow sources

- Wikidata main structured namespaces — CC0 согласно Wikimedia dump policy:
  <https://dumps.wikimedia.org/legal.html>.
- Russian/English Wikinews text после 2005-09-25 — CC BY 2.5 по той же официальной
  policy; нужен собственный dump extractor с article URL/attribution.
- arXiv full text — только item-level CC0/CC BY; стандартная arXiv distribution license
  не подходит. Metadata отдельно CC0:
  <https://info.arxiv.org/help/policies/submission_agreement.html>.
- Собственные детерминированные tool/dialogue traces — CC0 с versioned generator.

### Conditional

- Wikipedia/Wikimedia text: коммерческое использование разрешено, но CC BY-SA требует
  отдельного решения о ShareAlike для публичной модели. Conservative guidance Creative
  Commons прямо рекомендует учитывать SA для модели/outputs:
  <https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/>.
- YouTube Commons: CC BY и provenance выглядят полезно для разговорной речи, но нужны
  проверки transcript/translation quality и attribution export.
- Russian-PD: нужны OCR, archaic-language и jurisdiction audits.
- Russian law: допустимы только первичные тексты актов; NC-аннотации проекта исключаются.
- Project Gutenberg: public-domain вывод US-specific, плюс есть trademark/boilerplate
  условия и отдельные copyrighted items.
- The Stack: только актуальный opt-out-respecting release, per-file license и attribution.

### Исключено из production

OpenStax сейчас использует CC BY-NC-SA и прямо запрещает LLM training без разрешения:
<https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks>.
Текущие Russian/Ukrainian UD proxy также NC. Оба источника остаются research-only и не
могут случайно пройти production policy.

## 3. Streaming build

```bash
PYTHONPATH=src python scripts/build_corpus_shards.py imported-a.jsonl imported-b.jsonl \
  --output data/corpus-v1 \
  --registry configs/corpus/source_registry.json \
  --policy configs/corpus/policy_production.json \
  --protected eval/bilingual_smoke.json
```

Builder выполняет:

1. source/status/license/provenance gate;
2. NFKC normalization и quality/PII/secret filtering;
3. protected-eval contamination check;
4. exact SHA-256 и SQLite-backed SimHash/LSH near-dedup;
5. group-stable train/validation/test split;
6. deterministic timestamp-free JSONL.GZ shards;
7. per-shard hashes, rejection log и полный manifest.

SQLite index не держит все hashes/signatures в Python RAM. Shard reader проверяет как
compressed SHA-256, так и content hash каждой записи. Provenance bundle экспортируется
отдельно и связан с corpus hash:

```bash
PYTHONPATH=src python scripts/export_attribution.py data/corpus-v1 \
  --output data/corpus-v1-attribution.jsonl.gz
```

Частичный или упавший build намеренно не перезаписывается автоматически: output нужно
изучить и удалить явно, чтобы ошибка не превратилась в тихий resume с другим порядком.

## 4. Common Corpus import

После установки optional dependency:

```bash
pip install -e '.[data]'
PYTHONPATH=src python scripts/import_common_corpus.py \
  --output data/imported/common-corpus.jsonl \
  --acquisition-date 2026-08-20 \
  --languages en ru \
  --revision PINNED_DATASET_COMMIT
```

Oversized records режутся по paragraphs, но получают общий `split_group`, поэтому части
одного исходного документа всегда остаются в одном split. Ревизия обязана быть закреплена
для настоящего snapshot.

## 5. Tokenizer candidates и freeze

```bash
PYTHONPATH=src python scripts/train_tokenizer_candidates.py data/corpus-v1 \
  --vocab-sizes 16000 32000 48000 \
  --train-mib-per-language 256 \
  --eval-mib-per-language 16 \
  --output runs/tokenizer-candidates

PYTHONPATH=src python scripts/freeze_tokenizer.py \
  runs/tokenizer-candidates/report.json \
  --vocab-size 32000 \
  --output runs/tokenizer-v1
```

Sampling определяется hash документа и byte budget по языку, а не случайным состоянием
процесса. Frozen manifest связывает tokenizer SHA-256 с corpus SHA-256, policy, training
sample и special-token IDs. Выбор 32K остаётся гипотезой до этого измерения. После
freeze shards потоково упаковываются без materialization:

```bash
PYTHONPATH=src python scripts/pack_corpus_tokens.py data/corpus-v1 \
  --tokenizer runs/tokenizer-v1/tokenizer.json \
  --tokenizer-manifest runs/tokenizer-v1/manifest.json \
  --output data/tokens-v1
```

Каждый uint32 stream получает SHA-256 и связан одновременно с corpus и tokenizer hash.

## 6. Pinned GitHub real-data pilot

GitHub оказался стабильным источником первого реального snapshot. Конфигурация
`configs/corpus/github_pilot_sources.json` закрепляет три commit SHA:

| источник | revision | статус |
|---|---|---|
| OANC mirror, modern English | `0e93b4f` | OANC unrestricted use/redistribution |
| RusLit reviewed author folders | `4509680` | Public Domain |
| RusDraCor TEI | `b178c04` | CC0 |

```bash
PYTHONPATH=src python scripts/fetch_github_corpora.py \
  --output data/github-pilot-checkouts
PYTHONPATH=src python scripts/import_github_corpora.py \
  --checkouts data/github-pilot-checkouts \
  --output data/imported/github-pilot.jsonl \
  --acquisition-date 2026-08-20
```

Importer проверяет exact checkout SHA и clean worktree. RusLit ограничен двенадцатью
проверенными author directories; chunks одной книги получают общий split group.
RusDraCor извлекается из TEI с play-level provenance. Движущийся branch не используется.

После production policy, quality, PII, contamination и dedup gates:

| метрика | значение |
|---|---:|
| accepted documents | 8,560 |
| document text | 142.70 MB |
| English bytes | 67.99 MB (47.6%) |
| Russian bytes | 74.71 MB (52.4%) |
| OANC documents | 7,801 |
| RusLit chunks | 678 |
| RusDraCor chunks | 81 |
| corpus SHA-256 | `40f1191398b9ece1301f43279eec98c0c6d550ba305a721eb0ac78bd62ae3e4e` |

Главные отбраковки: 967 PII patterns, 298 repeated-line documents, 8 short, 4 near
duplicates, 2 encoding-damaged и 1 exact duplicate. Детерминированные accepted samples
вручную просмотрены: OANC содержит современную публицистику/travel/technical prose,
RusLit — чистую литературу, RusDraCor — корректно извлечённый диалог.

Tokenizer candidates на 128.4 MB balanced training sample:

| vocab | bytes/token | EN tokens/word | RU tokens/word | tied params @ d=512 |
|---:|---:|---:|---:|---:|
| 8K | 3.776 | 1.687 | 2.267 | 4.10M |
| 16K | 4.181 | 1.525 | 2.042 | 8.19M |
| 32K | 4.529 | 1.414 | 1.860 | 16.38M |

Для 10–30M L1-модели выбран **8K**: удвоение vocabulary до 16K съедает ещё 4.1M
параметров и удваивает LM-head cost ради умеренного сокращения последовательности.
Это решение только для scaling pilot; product tokenizer 32K остаётся открытым.

Packed result:

- train: **31,094,503 tokens**, 59.3% EN / 40.7% RU;
- validation: 1,178,019 tokens;
- test: 2,213,356 tokens.

Frozen tokenizer находится в `artifacts/tokenizer-github-pilot-v1`, полный manifest,
source revisions, shard hashes, candidates и token mixture — в
`results/github_pilot_data.json`.

Перед этим реальным прогоном synthetic vertical test также прошёл весь путь на 240
документах. Ограничение pilot: Russian часть преимущественно литературная/драматическая,
нет достаточных science, code, modern dialogue и tool данных. 31M train tokens годятся
для первого честного scaling checkpoint, но ещё не для полезного ассистента.

RUB GitHub corpus с 71,515 современными правительственными текстами найден и описан в
registry как `conditional`; в этот production-permissive snapshot он не включён.

## 7. Gate перед L1

Corpus v1 считается готовым только если:

- snapshot revision и все shard hashes закреплены;
- production policy не содержит conditional/research-only документов;
- RU/EN и domain token shares измерены после финального tokenizer;
- duplicate/PII/contamination rejection samples вручную просмотрены;
- attribution bundle экспортируется из manifest;
- 16K/32K/48K economics и bilingual smoke prompts заморожены до training.

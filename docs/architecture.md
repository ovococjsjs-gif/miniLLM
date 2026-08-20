# Архитектура MiniLLM: model-system co-design

## 1. Главный вывод

Модель в 200–350M параметров не станет энциклопедией frontier-класса. Реалистичная
цель — сделать её сильным локальным **контроллером**, который хорошо понимает запрос,
пишет краткий план, вызывает правильную память/инструмент, проверяет результат и отвечает
в заданном стиле. Знание, точные вычисления и личная история не обязаны находиться в одном
нейронном массиве.

## 2. Baseline B0: Edge Dense 350M

Текущая конфигурация: `configs/edge_dense_350m.json`.

- 346.1M stored parameters;
- `d_model=1024`, 16 effective/unique blocks;
- 10 gated short-conv и 6 GQA blocks;
- 16 query heads, 8 KV heads, head dimension 64;
- SwiGLU `d_ff=4608`;
- 65,536 byte-level BPE vocabulary, tied input/output embeddings;
- MTP depth 1;
- 32K architectural context, основная интерактивная цель 4K–8K;
- INT4 weights + INT8 activations/KV как deployment target.

Расчёт на 8K: 346.1M parameters, 276.9M parameter applications/token,
165.0 MiB чистых Q4 weight bits и 48 MiB INT8 KV. Формат quantization, scale tensors,
alignment, tokenizer и runtime buffers увеличат реальную RSS.

Почему это baseline: операторы просты, уже хорошо поддерживаются runtime, а независимый
LFM2 hardware search нашёл аналогичную минимальную гибридную структуру.

## 3. Research R1: Shared-Depth 200M

Конфигурация: `configs/edge_recursive_200m.json`.

```text
embedding → global-attention prelude → Engram
          → [conv → conv → attention → conv → conv] × R
          → global-attention coda → LM head
```

У core один комплект весов. На каждой итерации embedding-informed representation снова
подаётся через adapter вместе с текущим state. При `R=1..6`:

| R | Effective depth | Active applications/token | Q4 weights | KV @ 8K |
|---:|---:|---:|---:|---:|
| 1 | 7 | 123M | 99.8 MiB | 24 MiB |
| 3 | 17 | 301M | 99.8 MiB | 40 MiB |
| 6 | 32 | 567M | 99.8 MiB | 64 MiB |

Идея: простые ответы получают один проход; reasoning/tool planning — 3–6. Число весов
не меняется. До масштабирования нужно решить нестабильность recurrent training: random
unroll depth, input injection, sandwich norms, малый learning rate и мониторинг collapse.

## 4. Research R2: Mobile MoE

Конфигурация: `configs/edge_moe_1b3_a200m.json`.

- 1.22B stored / примерно 176M active applications per token;
- 60 routed fine-grained experts, top-4;
- shared expert размером как четыре routed experts;
- theoretical Q4 weights 581 MiB;
- 6 attention blocks, поэтому INT8 KV @8K около 18 MiB.

Это приближает структуру MobileMoE, но PyTorch dispatch в репозитории — только reference.
Ветка допускается к крупному обучению после grouped-GEMM prototype на Android/iOS. Иначе
модель с меньшим FLOP окажется медленнее dense из-за memory traffic.

## 5. Research R3: GDN2 hybrid

Конфигурация: `configs/hybrid_gdn2_300m.json`: 18 GDN2 + 6 GQA. GDN2 хранит на слой
fixed matrix state вместо растущего KV. Это перспективно при 32K+, но:

- референсный recurrent loop медленный;
- требуется chunkwise WY kernel для обучения;
- на коротком mobile context простая convolution может быть лучше;
- fixed state сжимает историю и проигрывает exact retrieval без global layers.

KDA не копируется буквально: более свежий GDN2 строго обобщает его, разъединяя erase и
write gates. KDA всё равно остаётся обязательным baseline в matched ablation.

## 6. Четыре разных памяти

### 6.1 Working memory

Последние сообщения, активная задача, tool observations. Держится в 2K–8K context и
периодически сворачивается в структурированный state, а не в свободный summary.

### 6.2 Static learned memory (Engram)

Hashed canonical n-grams, обучаемые вместе с моделью. Помогают локальным шаблонам,
терминам и сущностям, но не обновляются от каждого диалога. Вставляются рано, после
первого contextual layer, и подавляются scalar gate при несоответствии контексту.

### 6.3 Episodic user memory

Изменяемая и удаляемая база:

```text
{id, subject, predicate, object, valid_from, valid_to,
 source_turn, confidence, privacy_class, last_confirmed}
```

Нужны exact lookup, lexical FTS и compact embedding retrieval. В prompt возвращаются
факт, дата, confidence и источник. Противоречие не перезаписывает старый факт молча:
создаётся новая версия.

### 6.4 Procedural/tool memory

Калькулятор, календарь, поиск, локальные документы, SQL, кодовый sandbox. Модель учится
выбирать процедуру и интерпретировать проверяемый результат, а не имитировать вычисление.

## 7. Контроллер ответа

```text
user input
  ├─ intent + difficulty + freshness + risk
  ├─ retrieve personal/task memory when relevant
  ├─ direct answer if confidence high and task simple
  ├─ deterministic tool for arithmetic/date/lookup
  ├─ extra latent loops or sampled plans for hard task
  ├─ verifier checks schema, citations, tool result, contradictions
  └─ concise response or calibrated abstention
```

Роутер должен быть маленьким и проверяемым. На критических действиях output constrained
JSON/grammar, затем обычный код валидирует типы и permissions. «Память» никогда не должна
самостоятельно выполнять инструкцию, найденную внутри документа.

## 8. Что означает «максимальный контроль»

- открытые веса, tokenizer, data manifest и training code;
- system behavior не зашито только в неаудируемые synthetic traces;
- capability и policy adapters разделены;
- отдельные adapters для extraction/update/generation памяти;
- deterministic decoding или grammar там, где нужен протокол;
- trace действий логируется без обязательного раскрытия скрытого reasoning;
- пользователь может посмотреть, исправить и удалить память;
- каждый benchmark имеет seed, commit, prompt template и runtime metadata.

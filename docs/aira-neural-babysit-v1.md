# AIra Neural Babysit v1: коррекции меняют параметры, а не таблицу ответов

Статус: исследовательский gate пройден частично; production deployment **закрыт**.

## Исправление прежней формулировки

`SkillShelf` из AIra One — это полезный проверенный кэш ответов. Он ускоряет известные запросы и может защищать от уже наблюдавшихся опасных ошибок, но он не является обучением интеллекта: выбор по ключевым словам и возврат готового текста действительно эквивалентны аккуратно организованным `IF`.

Поэтому новый эксперимент полностью отключает shelf и проверяет другой контракт:

1. teacher correction изменяет числовые параметры;
2. при inference модели не передаётся готовый ответ;
3. отдельная переформулировка не участвует в оптимизации;
4. ответ генерируется авторегрессионно через Qwen с обученным neural residual;
5. незнакомые запросы проверяются отдельно, чтобы adapter не подменял их заученной темой.

## Что именно обучается

Qwen3.5-0.8B Q4 остаётся замороженным языковым donor: в доступных 3.8 GiB RAM полноценный backpropagation через 752M quantized parameters невозможен. Но теперь AIra содержит настоящий обучаемый компонент, влияющий на генерацию:

```text
Qwen hidden state, 1024
    ↓ normalize
MLP 1024 → 128 → 776 candidate-token logit deltas
    ↓ add to Qwen logits
next-token sampling
```

Отдельный learned gate `1024 → 32 → 1` решает, разрешено ли применять residual к данному запросу. Общий размер — **264,137 обучаемых параметров**. Все они оптимизированы совместно за 300 шагов.

`adapter.bin` содержит только числовые матрицы, bias, normalization statistics и token IDs. В нём нет task IDs, ключевых слов, готовых предложений или таблицы `prompt → answer`.

## Данные и отсутствие утечки

- 24 темы;
- две обучающие формулировки на тему: 48 records, 2,308 supervised next-token states;
- 24 исходные validation-переформулировки полностью исключены из optimization;
- 16 unrelated prompts обучают gate не вмешиваться;
- ещё 8 unrelated prompts остаются gate validation.

Для отдельного контроля все answer fields в inference TSV были заменены одним символом `-`. Все 24 neural outputs остались byte-identical исходному запуску. Значит native runtime не читает teacher answers при генерации. Результат: `results/aira_neural_adapter_independence_v1.json`.

## Результаты

### Teacher-forced next-token control

| held-out metric | frozen Qwen | Qwen + learned residual |
|---|---:|---:|
| NLL | 3.0236 | **0.0994** |
| exact full-vocabulary top-1 | 41.1% | **97.2%** |

Этот тест показывает, что adapter действительно выучил поправку к логитам, но teacher forcing сам по себе недостаточен: при свободной генерации ошибка одного токена меняет последующие hidden states.

### Полностью свободная генерация на 24 невиденных формулировках

| проверка | frozen Qwen | neural Babysit |
|---|---:|---:|
| keyword concept passes | 1/24 | 14/24 |
| после ручного просмотра | 1/24 | **13/24** |
| shelf/cache routes | 0 | 0 |

Ручная проверка отклонила один автоматический pass: ответ о фотосинтезе перечислил правильные понятия, но затем повторял одно предложение до лимита.

Успешные новые ответы включают сезоны, антибиотики, вакцины, парниковый эффект, фишинг, password manager, подозрительные ссылки, CO alarm, public Wi-Fi, RAM/SSD, compiler/interpreter, причины hallucination и SQL injection. Это новые формулировки, а не строки, совпадающие с train prompts.

### Out-of-scope preservation

Gate оставил frozen Qwen byte-exact на 6/8 новых unrelated prompts, но ошибочно активировал adapter для:

- антонима слова «горячий»;
- объяснения высыхания одежды.

Оба ответа были повреждены. Поэтому production gate остаётся закрытым.

## Что доказано, а что нет

Доказано:

- Babysit corrections могут изменять параметры AIra;
- изменённые параметры влияют на реальные Qwen logits;
- качество свободной генерации выросло на held-out paraphrases `1/24 → 13/24` после ручной проверки;
- при inference не читаются сохранённые teacher answers;
- это не keyword router и не ответ из shelf.

Не доказано:

- безопасная работа на произвольных запросах;
- отсутствие memorization внутри параметров;
- устойчивость за пределами 24 тем;
- достаточная autoregressive stability: 11/24 целевых ответов всё ещё провалены;
- готовность заменить полноценный LoRA/full-model update.

Любая маленькая модель может запоминать training trajectories. Существенное отличие этого эксперимента от `IF` — решение и последовательность токенов вычисляются непрерывными обученными функциями от hidden state, а перенос измеряется на не участвовавших в optimizer формулировках. Но только более широкие held-out и negative-control наборы покажут, насколько это обучение действительно обобщается.

## Следующий технический gate

1. добавить hard-negative gate curriculum и проверить на новом, не просмотренном control split;
2. перейти от ограничения в 776 candidate tokens к low-rank residual полного vocabulary;
3. обучать на собственных ошибочных rollout prefixes, а не только teacher-forced prefixes;
4. добавить repetition/KL regularization и generated-quality objective;
5. после прохождения этих проверок подключить adapter к AIra One; до этого `production_deployment_allowed=false`.

## Воспроизведение

```bash
python scripts/build_qwen35_output_adapter.py
python scripts/run_aira_neural_babysit.py
python scripts/audit_aira_neural_babysit.py
python scripts/verify_aira_neural_adapter_independence.py
```

Основные результаты:

- `results/aira_neural_babysit_v1.json`;
- `results/aira_neural_babysit_v1_audited.json`;
- `results/aira_neural_adapter_independence_v1.json`;
- `artifacts/aira-neural-babysit-v1/`.

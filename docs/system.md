# Локальная система ассистента

Маленькая модель используется как policy/controller, а не как единственный источник
знаний и вычислений. Runtime отделён от конкретного model provider и полностью тестируется
со scripted policy.

## Action protocol

Policy обязана вернуть ровно один JSON object:

```json
{"type":"tool_call","tool":"calculator","arguments":{"expression":"17*19"}}
```

или:

```json
{
  "type":"final",
  "content":"323",
  "confidence":1.0,
  "citations":["tool:0:calculator"],
  "memory_proposals":[]
}
```

Unknown fields, неизвестные tools, неправильные argument types и выдуманные citations
отклоняются runtime. Скрытое reasoning не является частью протокола.

## Permission model

Каждый tool требует отдельное permission:

- `compute`;
- `read_memory`;
- `read_documents`;
- `write_memory`;
- `network`.

Default agent получает первые три. Запись памяти и сеть по умолчанию запрещены. Tool
schema передаётся policy только тогда, когда permission действительно выдан.

## Реализованные tools

- безопасный decimal calculator через ограниченный AST без `eval`;
- ISO calendar: weekday, add days, date difference;
- temporal episodic-memory search;
- локальный SQLite FTS document search.

Calculator ограничивает длину/сложность expression и exponent. Calendar не угадывает
форматы дат, а требует ISO. Это намеренное уменьшение неоднозначности.

## Retrieval trust boundary

Все document chunks помечаются как `untrusted`. Runtime отдельно обнаруживает типичные
prompt-injection конструкции (`ignore previous`, `system prompt`, поддельные role tags) и
ставит `injection_warning`. Даже чистый документ остаётся данными, а не инструкцией.

Citation можно использовать только если она присутствовала в retrieved evidence или была
создана реальным tool event текущего trace. Несуществующая citation возвращает validation
error policy, а не попадает пользователю.

## Память

Persistent fact имеет:

```text
subject, predicate, object, valid_from, valid_to,
source_turn, confidence, privacy_class,
created_at, last_confirmed, superseded_by
```

Новый противоречащий факт создаёт temporal version и закрывает старый. Sensitive memory
не попадает в обычный retrieval. Модель может только предложить `memory_proposals`; Agent
не записывает их без отдельного пользовательского подтверждения.

## Bounded loop

Agent имеет жёсткий `max_steps`. Каждое действие и observation сохраняются в typed trace.
После превышения лимита возвращается безопасная ошибка с confidence 0, а не бесконечный
agent loop.

## Что ещё требуется

- LocalTorchPolicy с KV-cached generation;
- constrained JSON decoder вместо parse-after-generation;
- embedding retrieval рядом с FTS;
- подтверждение memory proposal в UI;
- sandboxed code tool;
- network search с domain allowlist и source provenance;
- adapters extraction/update/generation, обученные на trajectory datasets.

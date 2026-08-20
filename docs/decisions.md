# Decision log

## D-001 — Оптимизируем систему, не только backbone

**Решение:** считать retrieval, tools и persistent memory частью целевой архитектуры.

**Причина:** sub-billion model не может одновременно хранить всю энциклопедию, точно
считать, помнить пользователя и рассуждать как frontier model. Внешние проверяемые
примитивы дешевле и управляемее.

## D-002 — Conv/GQA dense является первым conventional control

**Статус:** понижен с главного пути до контрольной архитектуры решением D-010.

**Решение:** 10 gated short-conv + 6 GQA при ~350M.

**Причина:** LFM2 hardware search, доступность mobile kernels и первый local MQAR screen.
KDA/GDN2 и MoE сравниваются с ним, а не принимаются на веру.

## D-003 — Recurrent depth вместо безусловного роста весов

**Решение:** отдельный 209M shared-core candidate с 1–6 проходами.

**Причина:** на телефоне weight memory/bandwidth дефицитнее, чем короткий дополнительный
compute; сложность запроса должна управлять compute budget.

**Риск:** training collapse и высокая between-seed variance уже видны на toy proxy.

## D-004 — GDN2 предпочтительнее новой реализации KDA, но KDA остаётся baseline

**Решение:** reference code реализует более свежий Gated Delta Rule-2.

**Причина:** GDN2 строго содержит KDA как tied-gate special case и лучше в matched 1.3B
experiments. Для научной честности KDA kernel всё равно нужен в будущем сравнении.

## D-005 — Engram не равен памяти пользователя

**Решение:** learned static n-gram table и mutable episodic store проектируются отдельно.

**Причина:** пользователь должен исправлять/удалять факты; weights/hash table не дают
нужных provenance, temporal validity и access control.

## D-006 — INT4 QAT является target, ternary — эксперимент

**Решение:** production baseline использует стандартный INT4/INT8 путь.

**Причина:** зрелые runtimes и kernels. Native ternary потенциально лучше, но требует
отдельного pretraining и bitnet-like runtime.

## D-007 — Никаких больших runs без proxy scaling

**Решение:** минимум три model scales, три seeds, matched FLOPs/data и device profiling.
В текущей ограниченной CPU-среде отдельный test capped at 300 training steps.

**Причина:** ranking архитектур и data mixtures меняется с scale; single-run benchmark
создаёт дорогие ложные выводы.

## D-008 — Два baseline вместо преждевременного победителя

**Решение:** держать attention-only quality baseline и conv/GQA edge baseline параллельно.

**Причина:** при 300-step real-text proxy attention-only лучше по validation loss примерно
на 0.14, тогда как 2-attention/4-conv быстрее примерно на 15% в чистом forward и намного
лучше на generated MQAR. Inductive bias зависит от задачи, поэтому ни один вариант пока
не является универсальным победителем.

## D-009 — Trigger shelf не имеет права незаметно заменить policy

**Решение:** старый BPE n-gram shelf по умолчанию остаётся opt-in draft source. Hard
bypass разрешён только явно включённому AIra path после отдельной frozen/domain calibration
конкретного shelf/tokenizer и с burst/risk/anchor/cycle ограничителями; structured output и
policy-critical участки могут принудительно отключить bypass. Текущая transfer evidence
относится к byte/char shelf; CLI BPE archive — интеграционный эксперимент, не default.

**Причина:** старый BPE proxy дал лишь 2.52% coverage при 98.19% accuracy. Новый raw
trigger проходит отдельный transfer gate и действительно пропускает neural forward, но
cross-domain coverage остаётся небольшим. Heuristic bypass меняет распределение и поэтому
его teacher-forced, autonomous и end-to-end результаты всегда сообщаются отдельно.

## D-010 — AIra-v2 является главным исследовательским путём

**Решение:** целевая система — calibrated raw trigger → bounded episodic memory →
residual neural core → deep/tool escalation. Conventional Attention/Edge модели — matched
controls. Обычные большие scaling runs приостановлены до end-to-end cascade gate.

**Причина:** ещё одна маленькая decoder-only LLM не является отличимой инновацией проекта.
Аудит пользовательской AIra нашёл воспроизводимые trigger/memory/PC механизмы, но также
выявил neural starvation, ложные hybrid perplexity, O(ND) memory и незавершённую live
интеграцию. Канонические исправления и текущие измерения: `docs/aira-v2-audit.md`.

# Decision log

## D-001 — Оптимизируем систему, не только backbone

**Решение:** считать retrieval, tools и persistent memory частью целевой архитектуры.

**Причина:** sub-billion model не может одновременно хранить всю энциклопедию, точно
считать, помнить пользователя и рассуждать как frontier model. Внешние проверяемые
примитивы дешевле и управляемее.

## D-002 — Conv/GQA dense является первым baseline

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

**Причина:** ranking архитектур и data mixtures меняется с scale; single-run benchmark
создаёт дорогие ложные выводы.

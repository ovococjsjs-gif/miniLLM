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

## D-011 — Neural fallback является bounded byte-event core, а не exact cached LM

**Решение:** основной A3 runtime сохраняет raw-byte output. На fallback последние 64
байта детерминированно превращаются в dynamic BPE patches, bounded neural core предсказывает
один следующий байт и не хранит state, который нужно обновлять на shelf-позициях. Обычный
`MiniLLM` с KV catch-up остаётся correctness control.

**Причина:** exact catch-up лишь откладывает neural compute. Опрос byte shelf только на BPE
boundaries сохранил 1.28% coverage и разрушал canonical segmentation. Byte-event proxy на
каждой raw boundary улучшил proper validation perplexity на 3.0%, accuracy на 1.17 п.п. и
сократил neural calls на 2.61%, хотя Python runtime и автономное качество пока не прошли gate.

## D-012 — Mixer complexity не исправляет generated-context drift

**Решение:** gated 471K MLP остаётся event-core baseline. Parameter-matched conv и attention,
contiguous batches, 10% context corruption и четырёхбайтовый recovery rollout не допускаются
как новый default. Строгая UTF-8 mask включена как обязательный deterministic control.

**Причина:** 50/50 attention+recovery повышает autonomous accuracy с 7.40% до 7.80%,
но ухудшает static cascade ppl с 156.04 до 164.34, accuracy с 23.12% до 19.76% и требует
примерно в 18× больше training time. Conv получает 7.71% autonomous accuracy, но ухудшает
ppl до 172.90. Ни один из 18 runs не доказал safe generated-context threshold. Следующий эксперимент должен
учить устойчивый переход непосредственно на generated states или использовать более сильный
frozen/distilled fallback, а не добавлять ещё один локальный mixer.

## D-013 — Broad unique sampling принят, но не считается exposure решением

**Решение:** следующие event-core controls сэмплируют training examples минимум из
8M-token unique window вместо узкого 0.8M окна при том же лимите 300 updates.

**Причина:** matched broad-data control снижает cascade ppl с 156.04 до 149.11 и повышает
accuracy с 23.12% до 23.88% без дополнительных optimizer steps. Autonomous accuracy остаётся
7.38%, и generated-context threshold по-прежнему отсутствует. Это исправление data coverage,
не исправление переходной динамики.

## D-014 — Multi-byte head is the next acceleration gate; copy is not assumed multiplicative

**Решение:** первый base-training matrix проверяет 1/4/8-byte heads до масштабирования shelf и source-copy. Shelf/copy остаются отдельными calibrated actions; их theoretical coverage не перемножается с multi-byte compression. Predictable training positions сокращаются только через importance sampling с ненулевым unbiased control stream.

**Причина:** lossless event packing даёт oracle 8.00x для 8-byte labels, но текущая shelf копирует лишь 3.17% байтов и вместе с 8-byte head даёт 7.15x event compression, то есть дробит patches. Prompt-copy min-2 покрывает 42.06%, но снижает compression до 5.52x из-за коротких spans. При включённой shelf min-8/min-16 source copies дают 7.25x/7.24x event compression и не превосходят pure oracle 8-byte stream. Главный неизвестный — сможет ли multi-byte action model сохранить качество и пройти generated-context calibration.

## D-015 — Public Claude synthetic sets are quarantined; AI Babysit is on-policy and verifier-first

**Решение:** `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k` и `Roman1111111/claude-opus-4.6-10000x` не входят в weights до отдельного разрешения upstream service terms, source provenance и ручного quality audit. `WithinUsAI/claude_mythos_distilled_25k` отклонён как training target. AI Babysit принимается как собственный процесс только с legally usable teacher, exact student-prefix labels и deterministic verifier observations.

**Причина:** uploader Apache/MIT license не устраняет ограничения сервиса, которым были созданы outputs. Angrygiraffe не проверялся вручную и default config дублирует overlapping subsets; Roman содержит неверные category labels, benchmark contamination и encoding damage; Mythos set прямо не является output заявленного teacher и демонстрирует повторяющиеся templates. Babysit напрямую решает observed generated-state drift, но feedback должен превращаться в SFT/preference/process/KL targets, а не просто добавляться как текст.

## D-016 — Собственный synthetic seed называется AIra Mentor, а не имитирует Opus

**Решение:** проект выпускает `AIra Mentor v1` как CC0 verifier-first RU/EN SFT seed. Он содержит 6,000 conversations, 10 balanced categories и 23 deterministic template families. Внутренние reasoning traces не публикуются и не имитируются; targets состоят из коротких проверяемых объяснений, tool/memory actions, grounding, uncertainty и corrections. Dataset используется только после base pretraining с assistant-only loss.

**Причина:** собственный reproducible generator устраняет unclear upstream terms и позволяет привязать каждый target к verifier. При этом 0.7M tokens и шаблонная природа не могут создать Gemma-level base. Рост v2 разрешён через новые task families и реальные on-policy Babysit failures, а не простое размножение текущих templates.

## D-017 — Tiny random-init Mentor checkpoint is a failure collector, not the base

**Решение:** 1.7M `AIra Mentor Tiny v1` сохраняется только для локального interaction plumbing и AI Babysit rollouts. Он не продвигается в base model и не используется для quality claims, несмотря на validation ppl 2.43.

**Причина:** после 300 assistant-only steps teacher-forced NLL упал с 9.42 до 0.887, но strict autonomous verification прошёл лишь 1/10 category demonstrations. На 200 свежих seed-43 задачах прошли 7, все из memory-control; 193 failures стали первым Babysit correction set. Модель выучила формы ответов, но не научилась связывать новые числа, document IDs и code payloads. Base pretraining или pretrained initialization обязательны до полезного SFT.

## D-018 — Учитель является teacher-compiler, а локальная pretrained модель — только donor/control

**Решение:** компетентностную программу задаёт Arena.ai agent через failure clusters и `SkillPatch`; детерминированные solvers/verifiers размножают один разбор в свежий curriculum. Маленькие open checkpoints не получают статус учителя. Они могут передать языковой prior и служить baseline, но их ответы проходят те же verifiers и Babysit loop, что и ответы студента.

**Причина:** 4GB Gemma или sub-1B donor не имеют достаточной общей компетентности для роли арбитра. Одновременно массовое копирование красивых agent-ответов учит стиль, а не алгоритм. Первый Foundry pass свернул 193 ошибки в 11 причинных clusters и 11 patches, затем собрал 1,193 contrastive records: 1,000 новых детерминированных задач и 193 exact on-policy corrections. Matched 300-step tiny intervention снизил validation ppl 2.427 → 2.222, но оставил одинаковый fresh seed-45 strict score 0/10 → 0/10, подтверждая необходимость pretrained donor. Публичное weight use agent-authored patch text требует отдельной проверки Arena terms.

## D-019 — Qwen3.5-0.8B принят как структурный donor; state catch-up является hard gate

**Решение:** exact Unsloth Qwen3.5-0.8B Q4_K_M используется как первый pretrained language donor/control: 24 слоя, шесть групп `3×Gated DeltaNet + attention`, 752.4M parameter elements и 532.5MB artifact. В восстановленном GGUF нет отдельно именованных MTP tensors, поэтому MTP для этого файла не заявляется. AIra добавляет event routes, intermediate modes и state patcher; обычный wrapper не считается преобразованием архитектуры. Gemma E4B из активного плана удаляется.

**Причина:** Qwen donor помещается в 0.7–1GB бюджет и структурно ближе к recurrent/event программе, но measured capability недостаточна для роли teacher: balanced protected sample даёт 6/50 strict, 31/50 content и 0/18 required source attribution; fresh seed-46 rollout — 3/20 strict и 17 corrections. При этом runtime практичен: около 21.4 generated token/s и 852 MiB peak RSS на двух CPU threads. Пропуск recurrent groups без обновления будущего state некорректен. Поэтому `RecurrentStatePatcher` обучается против full-pass states и future logits, сохраняет attention anchors и обязан проходить generated-context calibration. Синтетический 200-step proxy получил MSE ratio 0.06075 и exact anchors. Real Qwen states теперь инструментированы, а projected held-out control получил MSE ratio 0.7421; injectible full-state replay всё ещё не выполнен.

## D-020 — Prompt-only protocol control rejected as an AIra quality fix

**Решение:** `aira-protocol-v1` сохраняется только как measured control и не считается исправлением Qwen donor. Source-pointer, tool execution и format routes должны стать проверяемыми runtime/architecture actions с fallback, а не длинным system prompt.

**Причина:** на тех же 20 fresh seed-46 задачах answer-free category hints подняли source fidelity с 0/7 до 3/7, content с 12/20 до 13/20 и protocol с 17/20 до 18/20. Но strict остался 3/20: оба tool calls получили правильную внешнюю JSON-схему, сохранив неверные arguments, а один memory answer выродился в список citations. Это полезное свидетельство controllability поверхности, но не прирост компетентности.

## D-021 — Qwen failures enter a fresh Foundry cycle without patch proliferation

**Решение:** 17 fresh seed-46 corrections компилируются с 1,000 новыми deterministic seed-47 contrasts в отдельный `aira-teacher-foundry-qwen-v1`. Protected 50-task baseline остаётся evaluation-only. Новый `SkillPatch` не добавляется, пока существующий catalog объясняет механизм.

**Причина:** Qwen packet даёт 11 clusters: operand binding, constraint binding, source identity, memory source/conflict, Python contract, tool schema и tool arguments. Все они маршрутизируются в существующие 11 patches. Итоговый curriculum содержит 1,017 records, включая ровно 17 on-policy corrections, SHA-256 `7e28525b0955efa63b763ae3a2ec6a43d32f2da5db3ec1514a21006eaa885de8` и явный protected-split count 0; генератор исключает все 6,000 известных Mentor v1 content hashes.

## D-022 — Real Qwen recurrent-state instrumentation gate passed; acceleration gate remains closed

**Решение:** pinned llama.cpp `cb_eval` probe становится каноническим источником real-state pairs. Он извлекает `state_predelta`, `new_state`, convolution cache, 24 layer outputs и full logits без fork/patch llama.cpp. Raw float32 остаются ignored; в Git входят source, build provenance, hashes и compact audit. Сам факт extraction не разрешает bypass.

**Причина:** для 18 Gated DeltaNet layers получены states `128×128×16` (1,048,576 bytes/layer). Между prompt stage и первым autoregressive event все 18 `new_state → state_predelta` и 18 `last_conv_states → conv_states` совпали byte-exact. Средний `||Δ||/||state_after||` следующего event равен 0.205614, максимум 0.611461, поэтому zero-update небезопасен. Но patcher ещё не обучен на этих real states, future-KL и generated-quality gate не пройдены, следовательно compute/quality acceleration не заявляется.

## D-023 — Projected real-state learnability passes, but only injectible replay can open bypass

**Решение:** `qwen35-real-state-pairs-v1` принимается как compact real-dynamics control: 48 transitions, disjoint prompt-group validation, recurrent+conv CountSketch state и probability-bucket future target. 69.6K patcher обучается state-first; future buckets остаются diagnostic. Этот результат разрешает разработку low-rank/full-state updater, но не layer skipping.

**Причина:** все 864 recurrent и 864 conv links прошли exact continuity до projection. На 16 held-out events copy-state MSE = 0.619321, mean-delta = 0.606894, learned patcher = 0.459573 (ratio 0.7421); cosine error снизился 0.244699 → 0.186590. Bucketed future KL также снизился 3.462925 → 3.334973, но coarse readout floor на true target state равен 3.298733. Projection lossy, state не инъецирован обратно в Qwen, generated quality не измерена — acceleration gate остаётся закрыт.

## D-024 — Partial-state replay sets a high true-vocabulary reconstruction target

**Решение:** `llama_state_seq_*_ext(..., PARTIAL_ONLY)` используется как fork-free injection gate. Attention KV и post-event position берутся из полного oracle event, после чего заменяются только recurrent+conv tensors. Copy/stale является обязательным нижним baseline; oracle interpolation — только sensitivity curve, не deployable method.

**Причина:** 12/12 full+partial serialization controls воспроизвели logits byte-exact. Stale recurrent state дал mean/median/max true-vocabulary KL `1.2945/1.0855/3.3657` и сохранил argmax лишь 8/12. При 25/50/75% oracle delta mean KL составил `0.4816/0.1533/0.00791`; даже 50% delta оставляет max KL 0.9633. Следовательно, bypass требует очень точного injectible updater и fallback; projected MSE improvement сам по себе недостаточен.

## D-025 — AIra One v0.1 ships as exact-event controller plus residual donor

**Решение:** первая запускаемая модель — `AIra One v0.1`: каждый запрос сначала проходит через high-precision event routes (tools, memory, documents, logic, code, arithmetic, installed Babysit skills), и только residual novelty вызывает Qwen. Fast/balanced/deep отличаются neural budget; unsafe recurrent-layer skipping выключен. Runtime предоставляет interactive CLI, OpenAI-compatible API, route statistics и append-only feedback journal.

**Причина:** Qwen baseline имеет язык, но strict control всего 6/50 и source 0/18. Exact AIra routes прошли 174/174 protected и 100/100 fresh Mentor-family records без neural calls. В первом mixed smoke два из трёх residual answers были плохими; on-policy teacher corrections стали двумя skills. Повторный smoke сократил neural calls 4 → 1, bypass вырос 2/5 → 4/5, суммарная request latency снизилась 32.37 → 2.71 s. Это измеримый первый end-to-end Babysit cycle, но не доказательство general intelligence: неизвестный residual всё ещё ограничен слабым 0.8B donor.

## D-026 — SkillShelf is a cache; Neural Babysit must change parameters

**Решение:** `SkillShelf` и маршруты `babysit.broad.*` ретроспективно классифицируются как reviewed answer cache, а не как обучение модели. Следующий AI Babysit gate требует числового parameter update, отключённого shelf при evaluation, disjoint prompt paraphrases и unrelated preservation controls. Первый `aira-neural-babysit-output-adapter-v1` обучает за 300 шагов 264,137 параметров prompt-gated `1024→128→776` hidden-to-logit residual поверх frozen Qwen.

**Причина:** пользователь справедливо указал, что возврат готового ответа после keyword match эквивалентен набору `IF` и не доказывает AI learning. В новом контроле teacher answers были доступны только при collection/training; inference с заменой всех answer fields одним `-` дал 24/24 byte-identical outputs. На 24 не участвовавших в optimizer paraphrases exact full-vocabulary teacher-forced top-1 вырос `0.4107→0.9723`, а вручную принятое free-generation quality — `1/24→13/24`, без shelf routes. Это доказывает parametric behavioral update, но deployment закрыт: 11/24 target prompts ещё провалены, а gate испортил 2/8 unrelated controls.

## D-027 — Active tree is reset to in-model AIra-Qwen work

**Решение:** stored-answer `SkillShelf`, broad answer-cache artifacts, hard-coded Babysit routes и candidate-token output-adapter v1 удаляются из active tree/package. AIra One остаётся controller для tools, memory, documents, feedback collection и Qwen fallback. Единственный активный neural-acceleration gate — learned full recurrent-state update с обратной инъекцией в Qwen; единственный допустимый Babysit gain — изменение checkpoint parameters с cache-free held-out generation.

**Причина:** answer lookup даёт полезный product cache, но не приближает основную модель к сочетанию «быстро + умно». Output-head residual менял параметры, однако не адаптировал внутреннюю архитектуру Qwen и провалил out-of-scope gate. Git history сохраняет эти отрицательные результаты без загрязнения текущего package. Сохраняются только необходимые foundations: pinned donor/runtime, real-state extraction, partial-state replay, correction records/verifiers и измерительные controls.

## D-028 — Stable Gated Delta parameters replace arbitrary full-state regression

**Решение:** первый active full-state updater предсказывает не SVD-факторы матрицы, а внутренние Qwen `key/value/gate/beta`, после чего применяет точную Gated Delta algebra. Кандидаты ограничены слоями `4/8/12/16/20` сразу после attention anchors. Patch strength `0.01` фиксируется только по train prompts; validation не участвует в calibration.

**Причина:** произвольный low-rank matrix predictor практически совпал с copy (`~1.000×`) и удалён как dead end. Stable-parameter model за 300 шагов получил held-out full-state MSE ratio `0.842777`. Public partial-state replay подтвердил byte-exact serialization controls и снизил validation mean true-vocabulary KL с candidate-copy `0.005605` до `0.003898`; oracle argmax сохранён `4/4` против `3/4`. Но один prompt регрессировал, alpha очень мал, conv new row остаётся oracle и реального skip ещё нет, поэтому generated-quality/speed/deployment gates закрыты.

## D-029 — Learned convolution cache removes the last oracle from full-cache replay

**Решение:** convolution cache для candidate layers обновляется как exact shift двух старых строк плюс learned newest-row residual. 2.42M-parameter updater обучается отдельно за 300 шагов и объединяется с Gated Delta state patch в формате `AIRASTP2`. Binding baseline оставляет stale и recurrent state, и convolution candidate layers; attention KV, position и остальные layers остаются matched oracle controls.

**Причина:** held-out newest-row MSE ratio равен `0.558072`. Без oracle convolution strict full-cache native replay по всем 16 validation transitions снижает mean true-vocabulary KL `0.021935 → 0.015856` (ratio `0.722852`), сохраняет oracle argmax `16/16` и улучшает 10/16 transitions. State-only alpha на расширенном наборе сам по себе не проходит mean gate. Шесть переходов регрессируют, свободная генерация и физический skip не выполнены — deployment и speed claims закрыты.

## D-030 — One hash-bound active pipeline replaces manual experiment assembly

**Решение:** `aira_qwen_active_v1.json` становится единственным operational contract. `run_aira_qwen_active.py` проверяет donor/runtime, build provenance, raw tensor inventory, exact split и normalization, затем запускает два bounded 300-step trainer, объединяет checkpoints, калибрует только на train и выполняет 16-transition validation replay. Combined checkpoint хранит component/config/source/normalization hashes и используется одним exporter path.

**Причина:** ручная последовательность команд допускала preprocessing drift и устаревшие cache artifacts. Tiny-corpus feature z-scoring был проверен и отклонён: он ухудшал held-out recurrent state; accepted normalization — internal per-sample LayerNorm, а train-only corpus statistics остаются audit-only hashes. Combined exporter дал byte-exact тот же `AIRASTP2`, что два принятых component checkpoints. После этой нормализации подготовка/сборка больше не является исследовательской переменной; остаются обучение, calibration и обязательные quality/speed gates.

## D-031 — Fixed scorecard prevents moving-target progress claims

**Решение:** headline фиксируется `aira_qwen_scorecard_v1`: те же 16 validation transitions, layers `4/8/12/16/20`, strict stale recurrent+conv baseline и alpha `0.01`. Train full-cache diagnostic optimum `0.05` сохраняется как диагностика, но не заменяет intervention без новой версии scorecard.

**Причина:** combined checkpoint был byte-identical компонентам, однако смена alpha `0.01→0.05` ухудшила held-out KL ratio `0.722852→0.829463`. Это была регрессия policy, а не модели, и прежняя подача смешивала разные test scopes. Теперь stage-1, train, projected и oracle-conv метрики не могут становиться headline. Все пять fixed metrics проходят текущие regression limits; новые модели сравниваются только с ними.

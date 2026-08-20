# Данные, обучение и distillation

## 1. Почему данные важнее экзотической архитектуры

Современные 1–3B модели обучаются на триллионах tokens. SmolLM2-1.7B использовал около
11T tokens и многостадийную смесь; LFM2 dense — 10T + 1T long-context mid-training;
MobileMoE — около 6T + 500B. Маленький backbone не компенсирует грязные дубликаты,
плохую языковую смесь или synthetic ответы без проверки.

Но начинать сразу с триллионов нельзя: сначала нужно построить scaling curves на малых
подвыборках и предсказать отдачу каждого домена.

## 2. Data pipeline

Для каждого документа сохраняются:

```text
source, license, acquisition_date, language, domain, quality_scores,
content_hash, near_duplicate_cluster, pii_flags, contamination_flags
```

Этапы:

1. license allowlist;
2. exact + MinHash/semantic near-dedup;
3. language/script identification;
4. PII и secrets filtering;
5. quality scoring отдельными classifiers для web/code/math/dialogue;
6. benchmark decontamination до tokenization;
7. document-aware packing с EOS и attention boundaries;
8. immutable manifests для каждого run.

Russian нельзя добавлять одним маленьким bucket в конце: tokenizer fertility и domain mix
оптимизируются вместе. Нужны современный web, книги, технические тексты, QA, code comments
и живой dialogue, но с отдельными eval holdouts по источникам и времени.

## 3. Mixture optimization

Вместо ручного «70/20/10»:

- обучить несколько 20–70M proxy models на разных смесях;
- измерять per-domain validation loss и target tasks;
- fit RegMix/data-mixing law;
- проверить предсказанную смесь на следующем scale;
- менять mixture только по заранее заданным capability bottlenecks.

High-quality data можно повторять, но repetition учитывается явно: overfit зависит от размера
модели, объёма уникального target domain и доли смеси.

## 4. Pretraining objectives

### Next-token prediction

Остаётся главным ground-truth objective. Loss считается по документным границам без утечки
между независимыми документами.

### Multi-Token Prediction

Один sequential MTP module получает hidden state и embedding следующего истинного token,
затем предсказывает token через один шаг. Это уплотняет supervision, учит представление
планировать вперёд и после отдельной настройки может стать draft head.

MTP loss не должен задавить main loss; стартовая гипотеза `λ=0.3`, затем sweep.

### Decoupled Top-K distillation

Для teacher сохраняются top-32 token IDs/logits и total probability mass. Objective разделён:

1. binary KL: совпадает ли масса student внутри teacher top-K;
2. conditional KL: совпадают ли относительные вероятности внутри top-K;
3. temperature применяется только ко второй части;
4. обычный hard-label CE остаётся.

Так storage teacher logits конечен, а truncated distribution не притворяется полным vocabulary.

## 5. Capacity-aligned post-training

Прямой перенос длинных traces от огромной reasoning-модели — плохой default. Для student
≤3B подтверждён learnability gap: короткие reasoning chains и умеренные teachers часто лучше.

Практический curriculum:

1. короткий правильный ответ и format following;
2. явная декомпозиция на 2–4 шага;
3. calculator/retrieval/code как действия;
4. observation grounding;
5. recovery после ошибочного tool call;
6. только затем длинные задачи.

Для каждой задачи генерируются несколько корректных trajectories. Предпочтение получает не
самая длинная или «красивая», а корректная trajectory с умеренным student NLL. Непосильные
traces либо упрощаются teacher, либо откладываются до более сильного checkpoint.

## 6. On-policy distillation и RL

Offline imitation не показывает student собственные ошибки. После стабильного SFT:

- student генерирует rollout;
- teacher получает тот же prefix плюс privileged answer/tool feedback;
- divergence считается на состояниях, куда реально пришёл student;
- шаги после грубой tool error down-weighted, чтобы cascade не давал ложную supervision;
- verifiable reward проверяет final answer/tests/schema.

RLVR используется для math/code/tool tasks. Открытые stylistic задачи идут через preference
optimization с length normalization; иначе модель учится многословию как proxy reward.

## 7. Optimizer

AdamW — контрольная точка. Muon заслуживает matched ablation: исследования до 4B сообщают
примерно 10–15% меньше tokens до одинакового loss и лучшую эффективность больших batch.
Но matrix parameters, embeddings, norms и router требуют разных update rules; hyperparameters
нельзя бездумно копировать с AdamW.

Обязательные логи: train/validation loss на tokens, FLOPs и wall-clock; gradient/update norms;
router load; activation outliers; hidden-token correlation для recurrent модели.

## 8. Quantization-aware training

Deployment target задаётся до pretraining. Для первого релиза:

- symmetric group-wise INT4 weights, group 32/64;
- dynamic INT8 activations;
- INT8 KV;
- FP16/FP32 norms и router при необходимости;
- sensitivity sweep для embeddings, LM head и FFN down-projection input.

QAT scaling work показывает, что quantization error зависит от model size, количества training
tokens и group size; больше pretraining не гарантирует, что финальный PTQ будет легче. Поэтому
QAT checkpoint и BF16 checkpoint оцениваются вместе на каждом late-stage milestone.

Native ternary — отдельный backbone, а не PTQ-флаг. Он проходит собственный scaling и только
потом сравнивается с INT4 по реальному joule/token.

## 9. Воспроизводимое продолжение runs

Checkpoint format v3 хранит model/optimizer, FP16 scaler, все RNG, генератор batch
sampling, step, validation state и run metadata. Resume допускается только при точном
совпадении model/train configs, corpus/tokenizer identity и сигнатур token streams; это
защищает от тихого продолжения на другой смеси данных или с другим schedule. Format-v2
FP32 checkpoints мигрируют с безопасными default-полями. CLI-пример:

```bash
PYTHONPATH=src python scripts/train_proxy_lm.py \
  --steps 300 --checkpoint-interval 50 --output runs/proxy

PYTHONPATH=src python scripts/train_proxy_lm.py \
  --steps 300 --checkpoint-interval 50 --output runs/proxy \
  --resume runs/proxy/step-150.pt
```

`steps` здесь остаётся исходной конечной целью schedule, а не числом дополнительных шагов.
Старые checkpoints без полного stochastic state намеренно не объявляются exact-resume.

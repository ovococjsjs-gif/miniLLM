# AIra → AIra-v2: technical audit and completion plan

Date: 2026-08-20. Audited source: user-owned
[`ovososjdjd-boop/AIra`](https://github.com/ovososjdjd-boop/AIra), branch
`arena/019fcef3-aira`, commit `1bc59758f653efab38e8a71b706a816854587a50`.
The owner explicitly permits reuse of AIra code and ideas in this project. This supersedes the
old license-based restriction in `docs/aira-review.md`.

## 1. Executive verdict

AIra was not disproved. It stopped between isolated successful mechanisms and an end-to-end
language substrate. Its strongest result is **not predictive coding alone**. It is the joint
principle:

> computation and learning should be paid for in proportion to novelty, while familiar
> patterns and episodic facts are handled by event-triggered memory.

Three mechanisms have enough evidence to carry forward:

1. a high-precision hierarchical trigger shelf and escalation;
2. bounded one-shot associative memory, read only on surprise;
3. local predictive-coding zones with coordinate sleep, but only where finite relaxation is
   demonstrably stable.

The original H-33 integration was promising but not a completed proof. On its templated M1
stream the hybrid beat the clean zone while training only 10.7% as many positions. However,
the neural arm was starved (`ppl=30.7`) and the shelf carried nearly the entire system. The
planned full live WikiText run (76k steps) was never completed; only a 2,400-step bridge exists.

AIra-v2 should therefore be a **triggered language substrate**, not a custom Transformer and
not a wholesale replacement of backpropagation on day one.

## 2. What actually worked

### 2.1 Local equilibrium credit

- EXP-01 verified that an equilibrated predictive-coding update aligns with BP at machine
  precision on linear and small tanh chains.
- Flat relaxation exhibits the predicted `O(L²)` credit diffusion.
- Jacobi preconditioning improved the constant by 16–23× but did not change `L²`.
- Multigrid reduced the measured depth exponent from about 2.00 to 0.11 in the linear setup.
- EXP-04 trained a 2-hidden-layer character LM: PC ended only 3.5% worse in perplexity than
  BP at equal weight-update count.

This agrees with the published result that predictive coding can approximate BP on arbitrary
computation graphs using local updates:
<https://doi.org/10.1162/neco_a_01497>.

### 2.2 Event sparsity inside relaxation

EXP-07 found a real coordinate-level structure: 84–94% of state-coordinate updates could be
suppressed with almost unchanged gradient alignment. This is stronger evidence than merely
thresholding ordinary feed-forward activations because the inactive coordinates were identified
inside a convergent solver and could reactivate.

### 2.3 Trigger shelf and escalation

On WikiText-2, the original fixed/cascaded character shelves reported approximately:

- 29.0% coverage at 95.0% top-1 accuracy for flat c8;
- 33.3% coverage at 94.6% for the hierarchy;
- hybrid perplexity 4.2–4.5% below the neural zone under the report's scoring rule.

CALC-03b improved the router from a confidence-only threshold to a two-dimensional
support/frequency gate. `(N>=2, f>=0.90)` reported 34.3% coverage at 95.66% accuracy.

H-33 on M1 demonstrated the core systems effect: the shelf+zone hybrid beat the clean zone
while only 10.7% of positions reached gradient updates. Even though M1 is unrealistically
repetitive, this is the clearest AIra mechanism worth transferring.

The direction is consistent with token-level deferral and speculative-cascade research, which
formalizes cost/quality routing rather than always invoking the expensive model:
<https://arxiv.org/abs/2405.19261>.

### 2.4 One-shot associative memory

- EXP-05 retrieved 12,800 synthetic episodes exactly with a unique cue.
- EXP-06 retained 16,698 natural-text sentence episodes with 0.998 cloze containment under
  heavy background and no meaningful age-related forgetting.
- EXP-09 showed that memory should be read on neural surprise: reading only 4–6% of positions
  improved the difficult masked subset substantially, while reading on every token hurt.

The basic dense-associative-memory direction is supported by modern Hopfield/DAM work, but
AIra's implementation is a brute-force nearest-neighbour scan, not a constant-time memory:
<https://arxiv.org/abs/1606.01164>.

### 2.5 Scaling laws discovered empirically

EXP-10–13 found real transfer rules for the small PC zone:

- relaxation depth should increase as output nudging `beta` decreases;
- learning rate scales approximately as width^-0.55 in the tested parameterization;
- lowering LR in the final third removed late drift;
- after tuning, the PC-vs-BP gaps at widths 96/256/512 were 1.66/2.88/5.64%.

These observations are compatible with later `muPC` work on stable width/depth
parameterizations for predictive coding: <https://arxiv.org/abs/2505.13124>.

## 3. What failed or remained unproved

### 3.1 Predictive coding was computationally expensive on dense hardware

EXP-04's PC step cost roughly 11× BP operations at `T=32`. The claimed advantage was local
weight residency and reduced communication, not fewer dense CPU operations. No calibrated watt
measurement was completed.

### 3.2 The recurrent language carrier failed

ZoneRNN reached a non-contractive state transition, relaxation stalled around residual
0.02–0.035, 74.5% of updates were rejected, and final perplexity was 2.35× BP. More solver
iterations did not fix systematic finite-equilibrium bias.

### 3.3 Sigma-delta only helped sparse traffic

On slowly varying synthetic channels sigma-delta compressed traffic 8–40×. On a saturated
inter-zone bus it was sometimes more expensive than dense FP16 because each event also needed an
address. The useful rule is therefore: event-code only already-sparse or slowly changing traffic.

### 3.4 Hard novelty filtering starved the validator

H-33's arm A removed gradients on shelf-covered positions. Its raw neural zone ended at
`ppl=30.7` versus 1.13 for the full-data control. The hybrid still scored well because its shelf
covered 91.9% of templated M1 positions. The neural validator then failed its precision gate.

This is the most important integration error. AIra-v2 must retain an all-token anchor through
soft weighting or a separately trained sentinel.

### 3.5 Multi-hop/local recurrence was not solved

The original recurrent PC path failed, and miniLLM's later R=1/2/4 shared-depth proxy also failed
to produce monotonic multi-hop accuracy. Repeating a block is not an algorithm without explicit
intermediate-state supervision or a learned transition objective.

### 3.6 Large-scale PC transfer remained open

At width 2,048, the best completed short run remained roughly 35% behind its BP control. Several
runs were unfinished. No 30M/100M local-learning language model was trained.

The newer PC-ALM result is directly relevant: layer-local dual variables change diffusive penalty
relaxation into faster primal-dual credit propagation and reportedly close the nonlinear BP gap at
finite inference budgets up to depth 128:
<https://arxiv.org/html/2605.31022v1>. AIra-v2 should test PC-ALM rather than continue tuning the
same quadratic-penalty solver indefinitely.

### 3.7 The full live H-33 experiment was never run

The repository contains M1 controls and 2,400-step live bridge checkpoints. It does not contain the
planned 76,000-step WikiText A/B pair. Therefore there is no completed end-to-end live-text proof of
H-33's training-efficiency claim.

## 4. Methodological and implementation errors to correct

1. **Static and online validation were mixed.** EXP-15/15b updated the shelf while scanning
   validation. This is valid continual adaptation, but it inflates a conventional frozen-heldout
   score. AIra-v2 reports frozen and online protocols separately.
2. **Hybrid perplexity was not always a normalized probability.** Some reports assigned
   `-log(conf)` on a shelf hit and `log(V)` on a miss; others floored unseen count probabilities
   without renormalizing. These are useful decision costs, not proper perplexity. AIra-v2 uses an
   explicit normalized Dirichlet/backoff distribution or reports cascade accuracy separately.
3. **DAM lookup was described as O(1), but code used `codes @ query`, O(ND).** Storage was
   compact relative to KV, but lookup still grew with episode count. AIra-v2 reports scan work and
   adds ANN/LSH only after recall tests.
4. **Python dictionary memory was underestimated.** Packed-record arithmetic (9 bytes/entry)
   was compared with an actual Python `dict[context, Counter]`, whose overhead is far larger.
   Both packed lower bound and resident RSS must be measured.
5. **Fixed update counts undertrained wider controls.** Larger clean zones becoming worse than
   smaller ones made the inferred capacity multiplier alpha unidentifiable. Future scaling uses
   matched tokens/FLOPs and converged controls.
6. **Multiple hyperparameters changed together.** Some beta/T/LR experiments mixed axes,
   making early conclusions ambiguous. One-factor interventions and preregistration stay mandatory.
7. **A sign bug existed in early PC feedback**, detected through negative gradient cosine. Finite
   differences and BP-alignment tests must be permanent unit tests.
8. **Shelf resume initially overwrote multi-continuation counts.** All stateful stores need
   roundtrip and interrupted-run equivalence tests.
9. **Character shelf and BPE neural model were never integrated.** AIra correctly predicted that
   BPE shelves would starve at small data, but the planned raw-byte-trigger/BPE-zone bridge was not
   built.
10. **No autonomous generation stress test existed.** Teacher-forced shelf accuracy does not catch
    self-reinforcing loops after one wrong bypass. Burst limits, cumulative risk, cycle guards and
    neural refresh points are required.

## 5. AIra-v2 architecture

```text
raw UTF-8 stream
    |
    +--> L0 frozen/online byte-char trigger hierarchy
    |       high calibrated confidence -> bounded burst
    |       otherwise -> event
    |
    +--> L1 bounded associative episodic memory
    |       familiarity + margin gate -> retrieved fact/state
    |       otherwise -> event
    |
    +--> L2 residual neural core (BPE/patch representation)
    |       direct token / tool / memory action
    |
    +--> L3 rare deep computation or deterministic tool
```

### Dual representation is intentional

The trigger operates on raw bytes/characters, where morphology and punctuation repeat. The neural
core operates on BPE or dynamic patches, where sequence length is affordable. Byte offsets bridge the
two streams. This implements the fallback already proposed in AIra ADR-01 after the BPE shelf failed.

### Training rule

The neural sentinel is initially trained on all tokens with ordinary BP. Residual specialization uses
soft target weights, never a zero-gradient mask:

```text
w_t = floor + (1-floor) * (1 - p_shelf(target_t))^gamma
```

A floor of 0.1–0.2 preserves language calibration while focusing most learning on surprising events.
Only after this passes a full-data control do we test local PC/PC-ALM updates on small adapters or
zones.

### Runtime rule

A trigger may bypass the neural core only under an explicit quality budget:

- support and calibrated confidence gate;
- maximum burst length;
- cumulative error-risk budget;
- repeated-cycle detector;
- periodic neural anchor;
- domain/OOD calibration;
- separate RU, EN, code and structured-output thresholds.

Lossless neural verification remains available as a control, but it cannot be the only mode because it
removes AIra's central compute-saving claim.

## 6. Implementation phases and death criteria

### A0 — frozen trigger transfer (current CPU)

Compare char60, UTF-8 bytes and 8K BPE on static RU/EN heldouts. Report coverage, accuracy, packed
memory and correct-burst distribution.

Gate: at least one raw representation reaches >=25% coverage at >=97% static accuracy, or provides
a clear cascade frontier that improves a proper neural mixture. Death: <10% coverage at 95% after
hierarchy and support calibration.

### A1 — soft residual learner (<=300-step proxies first)

Compare full LM loss, original hard filtering, and soft residual weighting. Quality gate: <=2%
proper hybrid-validation regression against full loss, with hard filtering retained as a negative
control. Mean sample weight is reported separately from actual dense training compute: a nonzero
control stream does **not** save backward compute without an event-sparse kernel, so the earlier
`<=40% effective gradient duty` wording was not a valid compute gate.

### A2 — bounded one-shot memory

Test exact facts, paraphrased cues, contradictions, unknown rejection, FIFO replacement and 1K–100K
capacity. Report O(ND) work honestly, then introduce LSH/ANN. Gate: known precision and unknown
rejection >=95% with provenance preserved.

### A3 — proper cascade

Measure frozen shelf -> memory -> neural core on the same stream. Primary metric is quality versus
actual neural calls and active bytes, not component-level perplexity.

### A4 — local continual learning

Implement PC-ALM and muPC parameterization first for 2–4-layer residual adapters. Compare BP, old
quadratic PC and PC-ALM at matched wall time, not matched weight steps. Gate: <=5% quality gap,
<=20% global-credit traffic and no more than 2x dense arithmetic before event-kernel claims.

### A5 — event runtime

Only after activity traces are measured, implement block-sparse CPU/Triton kernels. Sigma-delta is
allowed only where payload+address traffic beats dense transfer. Calibrate against RAPL/NVML.

## 7. Product claim if successful

AIra-v2 does not need to beat Gemma as a standalone encyclopedia. It must beat a conventional small
model on a different Pareto surface:

- fewer neural invocations per generated byte/token;
- one-shot, editable, provenance-aware local memory;
- bounded context state;
- local continual adaptation;
- deterministic tool/control behavior;
- measured latency and energy at matched output quality.

The existing miniLLM Attention model remains the control instrument. New ordinary scaling runs are
paused until an AIra-v2 component passes its small-scale gate.

## 8. Implemented and measured status (2026-08-20)

This section is the canonical status boundary between implemented evidence and the plan above.

### A0 trigger: passed the transfer gate, not the end-to-end gate

`src/minillm/aira/trigger.py` implements packed top-1 exact-count levels, hierarchical routing,
empirical/Wilson confidence, single-step inference and burst metrics. Frozen results and source
hashes are in `results/aira_trigger_proxy.json`.

- WikiText-2 UTF-8 empirical n5/p95: 31.46% coverage at 98.17% accuracy.
- WikiText-2 UTF-8 Wilson n2/p90: 22.86% at 98.82%.
- Russian same-book UTF-8 empirical n5/p95: about 27.3% at 97.2%.
- Russian cross-book UTF-8 empirical n5/p95 falls to 19.12% at 92.26%; the strict Wilson n5/p95
  point is 7.29% at 98.32%.

Therefore the trigger transfers, but raw count confidence is not domain confidence. A held-out
precision calibrator is implemented in `calibration.py`. In
`results/aira_calibration_proxy.json`, fitting on the first half of the cross-book text and testing
only on its second half gives UTF-8 10.36% coverage at 96.51% precision for a 95% target. The same
protocol gives 36.90% at 95.30% on in-domain WikiText-2. This is useful supervised domain
calibration, **not** an unlabeled OOD detector.

`generate_triggered_ids` now performs a real neural bypass and catches the model cache up after a
shelf burst. It enforces maximum burst, cumulative risk, periodic neural anchor and repeated-cycle
limits. The autonomous oracle-fallback stress in `results/aira_autonomous_proxy.json` evaluates
64,000 generated bytes per domain rather than teacher forcing every context:

- WikiText-2: 15.26% controlled bypass at 99.62% shelf precision;
- Russian same-book: 10.73% at 98.98%;
- Russian cross-book: 7.24% at 98.17%.

The controls rejected 689/233/37 candidate tokens respectively, but did not materially raise
precision in these short horizons. They remain necessary protection against rare loops, not a
claimed quality improvement. The fallback is an oracle, so these figures measure opportunity and
error propagation only—not language-model quality.

### A1 residual learner: quality gate passed, compute gate not established

`results/aira_residual_proxy.json` contains three seeds, identical initialization/sampling and 300
updates of a character context MLP. The shelf was frozen on a disjoint 1M-character prefix;
validation was also separate. The normalized top-1-plus-uniform-tail shelf distribution makes the
hybrid perplexity proper under the stored information.

| rule | mean sample weight | neural ppl | hybrid ppl | hybrid accuracy |
|---|---:|---:|---:|---:|
| full | 1.000 | 10.323 | 9.099 | 40.52% |
| hard filter | 0.874 | 10.820 | 9.219 | 40.25% |
| soft residual, floor 0.15 | 0.895 | 10.630 | 9.165 | 40.35% |

Soft residual recovers roughly half the hard-filter regression and stays within 0.73% hybrid
perplexity of full loss, so it repairs starvation at this scale. Full loss still wins. Mean weight
0.895 is not an 89.5% wall-time claim; all three execute dense backward passes. Residual weighting
is accepted as a safe specialization rule, not yet as a training-speed result.

`mixture.py` separately implements normalized shelf-only distributions and convex shelf/neural
probability mixtures with endpoint and gradient tests. Pseudo-perplexity from mixing confidence
scores is no longer allowed.

### A2 memory: random-code mechanics passed; semantics and indexing open

`memory.py` implements bounded ring storage, familiarity/margin rejection, conflict rejection,
provenance, deletion and explicit linear scan accounting. In `results/aira_memory_proxy.json`, 512-D
random bipolar codes recall 100% of sampled facts through 30% bit flips for 100, 1,000 and 5,000
facts; 40% noise and random unknowns are rejected. At 5,000 facts, code storage is 2.56 MB and mean
known-query latency is about 1.4 ms on this host (timings vary between runs).

These are independent random codes. No learned RU/EN semantic encoder, paraphrase benchmark or
sublinear phone-scale index has passed yet.

### A3 cascade: first vertical slice found and repaired the representation/state error

The initial `MiniLLM` integration skips a forward call when a shelf token is chosen and later catches
the KV/state cache up on the emitted suffix. This is an exact integration control, but it does **not**
save all neural layer execution: catch-up eventually processes those tokens. The earlier wording
calling it a true active-compute bypass was incorrect. The tokenizer-bound archive, CLI telemetry and
cache-equivalence tests remain useful controls.

`results/aira_token_bridge_proxy.json` tests the next obvious bridge: query the raw-byte shelf only at
canonical BPE-token boundaries and emit the longest vocabulary token matching predicted bytes. It
fails as a primary bridge:

- only 1.28% of sampled BPE boundaries route, despite a 1.89M-byte shelf;
- canonical-token precision is 15.63%, while byte-prefix precision is 96.88%;
- the oracle-fallback autonomous run routes only 2.15% of events;
- the weak 300-step BPE core leaves the shelf manifold immediately and routes zero autonomous events.

Most useful byte-shelf matches occur *inside* BPE tokens. Requiring a canonical token boundary loses
the mechanism, while accepting a shorter valid token changes segmentation and destabilizes the
ordinary token LM.

The implemented repair is `ByteEventLM` plus `generate_byte_events`:

1. generation and shelf routing remain raw-byte aligned;
2. only a neural fallback dynamically BPE-merges the latest bounded 64-byte window;
3. the neural core reads 16 dynamic BPE patches and predicts one raw byte;
4. no recurrent/KV state is advanced through shelf bytes, so each accepted byte is a genuine skipped
   neural invocation;
5. `AIraCascade` serves an explicitly requested, accepted structured episodic fact directly with
   provenance; unknown/conflicted keys fall through to shelf→neural generation, and deletion remains
   user-controlled.

`results/aira_byte_event_proxy.json` trains the 471,232-parameter FP32 event core for 300 steps over
three seeds. On 10,000 independently sampled validation bytes, the proper normalized cascade gives:

| metric | neural only | byte-event cascade |
|---|---:|---:|
| mean perplexity | 160.87 | **156.04** |
| mean accuracy | 21.95% | **23.12%** |
| neural call fraction | 100% | **97.39%** |
| FP32 parameter-read proxy | 1.885 MB/byte | **1.836 MB/byte** |

The shelf's routed-byte precision is 95.40%. Thus the first teacher-forced vertical slice improves
quality while reducing neural calls and batch-1 parameter traffic by 2.61%; this is the first A3
component result that satisfies the *direction* of quality-adjusted active compute.

It does not yet pass the product/runtime gate. Unfused Python dynamic BPE+shelf routing makes the
cascade validation path about 50x slower than the batched neural evaluation. In autonomous 64-byte
generation, the undertrained core has only 6.68% byte accuracy and quickly leaves the familiar
manifold. The uncalibrated cascade routes just 0.33% of bytes with 35.97% mean shelf precision. On a
separate 100-sequence generated-context calibration split, candidate precision is only 7–8% for every
seed and no threshold can prove 95% precision. The calibrated policy therefore disables autonomous
bypass and exactly restores the neural-only quality/call count. By contrast, oracle fallback retains
99.88% byte accuracy, 2.67% shelf routing and 95.32% autonomous shelf precision. This confirms both
that fallback quality is the limiting factor and that generated-context calibration must be a hard
safety gate rather than an optional report. `AIraCascade` consequently disables autonomous shelf
routing unless it receives a fitted non-empty `ReliabilityThreshold`. The next gate is a stronger
matched event core and fused
incremental BPE/hash lookup; no general speedup claim is justified yet.

### A4 PC-ALM: mathematical reference supports the repair, not an efficiency claim

`pc_alm.py` follows the PC-ALM primal/dual schedule with autograd used only as a mathematical
activity-gradient reference. `results/pc_alm_proxy.json` compares old finite PC and PC-ALM against
BP over three seeds. At depth 8 and `T=2L`, global cosine rises from 0.306 to 0.814; at `T=8L`,
PC-ALM reaches 0.982. At depth 16, PC-ALM reaches 0.744 at `2L` and 0.954 at `8L`, while minimum
per-layer cosine is only 0.313 and 0.634 respectively. Same-T reference runtime is about 1.3x old
PC.

This validates primal-dual credit propagation as the branch to pursue, but does not reproduce the
paper's deep `2L` result under this naive tanh parameterization and does not provide a local kernel.
The next valid experiment is a residual/Depth-μP adapter at matched wall time, not a larger ordinary
LM run.

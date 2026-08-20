# Small-scale experimental architecture diagnostics

Status: CPU-only mathematical/runtime probes, 2026-08-20. Every training run is capped at
300 steps. These results select implementation work; they are not language-quality or
phone-latency claims.

## 1. Sequence mixer scaling

`python scripts/benchmark_sequence_mixers.py` compares bare mixers at `d=64`, batch 4 and
sequence lengths 16–256. At length 256:

| mixer | parameters | forward+backward tokens/s | FP32 decode state | theoretical sequence term |
|---|---:|---:|---:|---|
| GQA attention | 12,320 | 126,278 | 256 KiB | `O(T² H D)` |
| gated short conv | 12,544 | 317,733 | 2 KiB | `O(T D K)` |
| GDN2 reference | 32,912 | 5,781 | 16 KiB | `O(T H D_head²)` |

The convolution is parameter-matched to attention, 2.52× faster in this long end of the
CPU microbenchmark, and its state is 128× smaller. Attention state grows linearly with
context; convolution and GDN2 state stay fixed.

GDN2 has the desired fixed-state mathematics, but the sequential correctness reference is
21.8× slower than optimized PyTorch attention and 55.0× slower than convolution here. It
also has 2.67× as many projection parameters as the attention mixer. Therefore a real GDN2
training run is blocked on a chunkwise/fused kernel; scaling the Python loop would measure
implementation overhead, not the architecture.

The empirical time exponents in the JSON are not asymptotic complexity estimates: dense
projection cost, launch overhead and fused SDPA dominate these tiny lengths. The report
also records transparent operator-only work formulas.

## 2. Adaptive recurrent depth on associative recall

`python scripts/run_adaptive_depth_proxy.py --steps 300` trains three 124K-parameter
shared-core variants for three seeds:

| training rule | R=1 accuracy | R=2 | R=4 | mean train seconds |
|---|---:|---:|---:|---:|
| random unroll | 28.87% | 28.22% | 29.36% | 11.87 |
| + learned step embedding | 29.30% | 28.13% | 28.45% | 11.89 |
| + deep/shallow sandwich KL | 29.62% | 27.86% | 27.86% | 25.66 |

Chance is 3.125%. Learned iteration embeddings cost only 256 parameters and preserve exact
cached decoding, but they do not create a reliable quality increase with depth. Sandwich
self-distillation slightly improves the fast exit while making training 2.16× slower and
does not improve the deep exit. Naive extra recurrences therefore do not yet implement a
useful `deep` mode.

Runtime does scale as intended: mean forward throughput for random unroll is 164.8K,
99.9K and 58.4K tokens/s at R=1/2/4. R=4 costs about 2.82× R=1 rather than loading another
weight set, but no consistent quality gain currently pays for that compute.

## 3. Iterative composition stress test

`python scripts/run_pointer_chase_depth_proxy.py` encodes one random 8-node cycle and asks
for its 1–4 hop composition. A curriculum spends 50% of 300 steps on one hop, then exposes
2/3/4 hops. Chance is 12.5%.

| training rule | one-hop R=1 | mean fast | mean deep | train seconds |
|---|---:|---:|---:|---:|
| random unroll | 50.07% | 21.56% | 21.67% | 12.02 |
| + learned step embedding | 71.52% | 27.82% | 26.67% | 12.11 |
| + sandwich KL | 82.32% | 29.43% | 28.96% | 26.74 |

Step conditioning and sandwich supervision materially improve one-hop retrieval. However,
all variants remain near chance on two-to-four-hop composition, and R=4 does not beat R=1
on average. The model learned retrieval/copying, not a reusable iterative algorithm. An
earlier random-permutation version was rejected because four-hop cycle shortcuts made the
task biased; the committed v2 uses one full cycle.

## 4. Decisions

1. **Keep convolution in the active branch.** It has a clear small-scale speed/state win.
2. **Keep step conditioning implemented but disabled by default.** It is mathematically
   cheap and useful for one-hop specialization, but has not passed the deep-compute gate.
3. **Do not promote sandwich consistency to real-data training.** Current benefit is mostly
   fast-exit copying at more than twice training cost.
4. **Do not train reference GDN2 at scale.** Implement and verify a chunkwise kernel first.
5. **Redesign adaptive depth around explicit iterative state supervision.** The next proxy
   must require each recurrence to produce a verifiable intermediate pointer/state, then
   test monotonic accuracy at R=1/2/4 and learned halting.
6. **Use better data as fuel, not as the project identity.** A normal Attention model on
   Data v2 remains the control; experimental candidates must beat it at matched storage and
   measured compute.

Machine-readable evidence is in `results/sequence_mixer_benchmark.json`,
`results/adaptive_depth_proxy.json`, and `results/pointer_chase_depth_proxy.json`.

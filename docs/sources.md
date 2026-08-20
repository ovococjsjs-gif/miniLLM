# Аннотированные первоисточники

Ссылки ведут преимущественно на arXiv/официальные страницы. Все числа в design docs
нужно сверять с конкретной версией статьи и target runtime.

## Архитектура и mobile

1. **MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases**
   (ICML 2024) — deep-thin, tied embeddings, GQA, immediate block-wise sharing.
   <https://proceedings.mlr.press/v235/liu24ce.html>

2. **LFM2 Technical Report** (2025) — hardware-in-the-loop search; gated short conv +
   небольшое число GQA; 350M–2.6B dense и 8B-A1B MoE; device throughput.
   <https://arxiv.org/abs/2511.23404>

3. **MobileMoE: Scaling On-Device Mixture of Experts** (2026) — 0.3–0.9B active,
   fine-grained/shared experts, INT4 QAT и реальные S25/iPhone 16 Pro measurements.
   <https://arxiv.org/abs/2605.27358>

4. **A Systematic Evaluation of On-Device LLMs: Quantization, Performance, and Resources**
   — capability + latency + resource methodology и сильная деградация слишком низких bits.
   <https://arxiv.org/abs/2505.15030>

5. **PalmBench** (ICLR 2025) — автоматизация mobile latency, memory, power, quality и
   harmful-output measurements.
   <https://openreview.net/forum?id=0Qe6obx2kl>

## Linear/recurrent attention

6. **Kimi Linear: An Expressive, Efficient Attention Architecture** — KDA, 3:1 KDA/MLA,
   48B-A3B matched experiments, kernels и checkpoints.
   <https://arxiv.org/abs/2510.26692>

7. **Gated Delta Networks: Improving Mamba2 with Delta Rule** — основа gated delta
   recurrence и hybrid с sliding-window attention.
   <https://arxiv.org/abs/2412.06464>

8. **Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention** — channel-wise
   erase/write, matched 1.3B/100B comparison с KDA и Mamba-3.
   <https://arxiv.org/abs/2605.22791>

9. **Qwen3-Next / Qwen3.5 model architecture** — production validation 3:1 Gated
   DeltaNet/global attention, включая small checkpoints.
   <https://huggingface.co/Qwen/Qwen3.5-0.8B>

## Parameter sharing и latent compute

10. **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach**
    — prelude/recurrent core/coda, random unroll, input injection, collapse diagnostics.
    <https://arxiv.org/abs/2502.05171>

11. **Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level
    Computation** — token-level depth routing, recursion-wise/shared KV, 135M–1.7B runs.
    <https://arxiv.org/abs/2507.10524>

## Conditional и persistent memory

12. **Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language
    Models** — DeepSeek Engram, canonical n-grams, multi-head hash, contextual gate,
    20–25% allocation optimum.
    <https://arxiv.org/abs/2601.07372>

13. **Benchmarking and Enhancing Long-Term Memory in LLMs** — episodic + working +
    scratchpad и тесты до 10M-token conversation histories.
    <https://arxiv.org/abs/2510.27246>

14. **MemLoRA: Distilling Expert Adapters for On-Device Memory Systems** — отдельные
    adapters extraction/update/generation для локальной памяти.
    <https://arxiv.org/abs/2512.04763>

## DeepSeek

15. **DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence** —
    CSA/HCA, mHC, DeepSeekMoE, MTP, Muon, Flash/Pro configs.
    <https://arxiv.org/abs/2606.19348>

16. **Официальный DeepSeek V4 Preview release** — подтверждает 284B/13B Flash,
    1.6T/49B Pro, 1M context и open weights.
    <https://api-docs.deepseek.com/news/news260424/>

17. **DeepSeek-V3 Technical Report** — MLA, DeepSeekMoE, loss-free balancing и
    sequential MTP; reported 1.8× TPS от accepted draft token.
    <https://arxiv.org/abs/2412.19437>

## Quantization и tokenization

18. **BitNet b1.58 2B4T Technical Report** — native ternary weights, 2B/4T и
    bitnet.cpp deployment measurements.
    <https://arxiv.org/abs/2504.12285>

19. **Bitnet.cpp: Efficient Edge Inference for Ternary LLMs** — TL/I2_S kernels и
    CPU/ARM measurements.
    <https://arxiv.org/abs/2502.11880>

20. **Scaling Law for Quantization-Aware Training** — 268 W4A4 runs; зависимость
    quantization error от N, D и group size; FC2 input outliers.
    <https://arxiv.org/abs/2505.14302>

21. **Byte Latent Transformer: Patches Scale Better Than Tokens** — dynamic entropy
    patching, robustness и scaling до 8B/4T bytes.
    <https://arxiv.org/abs/2412.09871>

22. **Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies** —
    совместное масштабирование vocabulary и non-vocabulary parameters.
    <https://arxiv.org/abs/2407.13623>

## Данные и optimizer

23. **SmolLM2: When Smol Goes Big** — полностью открытая 1.7B/11T data-centric recipe,
    multi-stage mixture, FineMath/Stack-Edu/SmolTalk.
    <https://arxiv.org/abs/2502.02737>

24. **Data Mixing Laws** — прогноз mixture performance по small-scale runs.
    <https://arxiv.org/abs/2403.16952>

25. **RegMix: Data Mixture as Regression** — параллельные short proxy runs для поиска
    смеси данных.
    <https://arxiv.org/abs/2407.01492>

26. **Practical Efficiency of Muon for Pretraining** — matched compute/time сравнение
    до 4B; reported 10–15% token efficiency gain против AdamW.
    <https://arxiv.org/abs/2505.02222>

## Distillation, reasoning и tools

27. **MiniLLM: On-Policy Distillation of Large Language Models** — reverse-KL и
    on-policy generative distillation. Не связан с кодом этого репозитория.
    <https://arxiv.org/abs/2306.08543>

28. **Small Models Struggle to Learn from Strong Reasoners** — learnability gap для
    ≤3B, short/long и small/large-teacher trace ablations.
    <https://arxiv.org/abs/2502.12143>

29. **Distilling LLM Agent into Small Models with Retrieval and Code Tools** — 0.5B,
    1.5B, 3B students, first-thought prefix и self-consistent actions.
    <https://arxiv.org/abs/2505.17612>

30. **On-Policy Context Distillation for Language Models** — reverse-KL на student
    trajectories, experiential/system-prompt distillation.
    <https://arxiv.org/abs/2602.12275>

31. **SOD: Step-wise On-policy Distillation for Small Language Model Agents** —
    down-weighting teacher signal после tool-induced state divergence.
    <https://arxiv.org/abs/2605.07725>

## Evaluation

32. **ThinkSLM** — reasoning, intermediate errors, GSM-Plus perturbations и сравнение
    evaluation methods для SLM.
    <https://arxiv.org/abs/2502.11569>

33. **RUPBench** — 15 reasoning datasets и 9 видов семантически сохраняющих perturbations.
    <https://arxiv.org/abs/2406.11020>

34. **BeyondBench** — динамически генерируемые алгоритмические задачи с детерминированной
    проверкой, снижающие риск benchmark memorization.
    <https://arxiv.org/abs/2509.24210>

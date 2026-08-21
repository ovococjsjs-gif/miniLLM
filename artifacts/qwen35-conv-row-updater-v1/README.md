# Qwen3.5 convolution-row updater v1

Predicts the only non-deterministic newest row of the three-row convolution cache for recurrent layers 4, 8, 12, 16 and 20. The two older rows are shifted exactly.

- parameters: 2,422,001;
- optimizer steps: 300;
- held-out newest-row MSE ratio vs repeating the previous newest row: 0.558072;
- combined with the recurrent-state updater, strict full-cache injected KL improves from 0.021935 to 0.015856 over 16 held-out transitions;
- full learned patch preserves oracle argmax on 16/16 transitions and improves KL on 10/16.

Six transitions still regress and free-generation/speed gates remain closed.

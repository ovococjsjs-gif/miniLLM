# Qwen3.5 projected real-state patcher v1

Local checkpoint location for the 69,569-parameter projected-state patcher and 43,410-parameter future-bucket readout. `model.pt` is intentionally ignored; exact run metrics and its checkpoint SHA-256 are stored in `results/qwen35_real_state_patcher_proxy.json`.

The checkpoint operates on lossy 80-dimensional per-layer sketches. It cannot be injected into llama.cpp’s full `128×128×16` states and must not be distributed or described as an accelerated Qwen model.

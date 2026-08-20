"""Static parameter, memory, and decode-cost accounting.

The estimates are transparent proxies, not benchmark claims. Wall-clock performance
must always be measured on the target phone/runtime because sparse dispatch, kernel
fusion, cache hierarchy, and thermals dominate simple FLOP counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import MiniLLMConfig, layer_type_counts


@dataclass(frozen=True)
class ModelProfile:
    unique_parameters: int
    engram_parameters: int
    unique_backbone_parameters: int
    active_parameter_applications_per_token: int
    effective_depth: int
    unique_depth: int
    weight_memory_bytes: int
    kv_cache_bytes: int
    recurrent_state_bytes: int
    approximate_decode_flops: int
    layer_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mixer_params(config: MiniLLMConfig, kind: str) -> int:
    d, h, kv, hd = config.d_model, config.n_heads, config.n_kv_heads, config.head_dim
    if kind == "attention":
        return d * (h * hd) + 2 * d * (kv * hd) + (h * hd) * d + 2 * hd
    if kind == "conv":
        return 2 * d * d + d * config.conv_kernel + d + d * d
    if kind == "gdn2":
        # Eight dense d×d projections, per-head RMS scale, and two d-wide
        # decay parameter vectors.
        return 8 * d * d + 2 * d + hd
    raise ValueError(kind)


def _ffn_params(config: MiniLLMConfig, *, active: bool) -> int:
    d = config.d_model
    if not config.moe.enabled:
        return 3 * d * config.ffn_hidden
    moe = config.moe
    expert_count = moe.top_k if active else moe.num_experts
    return (
        d * moe.num_experts
        + expert_count * 3 * d * moe.expert_hidden
        + (3 * d * moe.shared_expert_hidden if moe.shared_expert_hidden else 0)
    )


def _block_params(config: MiniLLMConfig, kind: str, *, active: bool) -> int:
    norms = (4 if config.sandwich_norm else 2) * config.d_model
    return _mixer_params(config, kind) + _ffn_params(config, active=active) + norms


def _engram_params(config: MiniLLMConfig) -> int:
    if not config.engram.enabled:
        return 0
    e = config.engram
    tables = len(e.ngram_orders) * e.num_hash_heads * e.table_size * e.embedding_dim
    fusion = 2 * e.retrieved_dim * config.d_model
    refine = config.d_model + config.d_model * e.conv_kernel
    return tables + fusion + refine


def profile_model(
    config: MiniLLMConfig,
    *,
    context_length: int = 4096,
    weight_bits: int = 4,
    kv_bits: int = 8,
    state_bits: int = 16,
    core_repetitions: int | None = None,
) -> ModelProfile:
    config.validate()
    repeats = config.core_repetitions if core_repetitions is None else core_repetitions
    effective = config.effective_layer_types(repeats)
    unique = config.prelude_layers + config.core_layers + config.coda_layers
    d = config.d_model

    embedding = config.vocab_size * d
    untied_head = 0 if config.tie_embeddings else embedding
    unique_blocks = sum(_block_params(config, kind, active=False) for kind in unique)
    adapter = (2 * d * d + d + d) if config.recurrent_input_injection else 0
    step_conditioning = (
        config.max_core_repetitions * d if config.recurrent_step_conditioning else 0
    )
    final_norm = d
    mtp = config.mtp_depth * (2 * d * d + d)
    engram = _engram_params(config)
    total = (
        embedding
        + untied_head
        + unique_blocks
        + adapter
        + step_conditioning
        + final_norm
        + mtp
        + engram
    )

    # Matrix/vector weights touched for one generated token across effective depth.
    # The recurrent adapter is reused (and therefore applied) once per core pass.
    active_blocks = sum(_block_params(config, kind, active=True) for kind in effective)
    active_adapter = (
        repeats * (2 * d * d + d) if config.recurrent_input_injection else 0
    )
    active_step_conditioning = repeats * d if config.recurrent_step_conditioning else 0
    active_engram = 0
    if config.engram.enabled:
        e = config.engram
        active_engram = 2 * e.retrieved_dim * d + d * e.conv_kernel + d
    active = active_blocks + active_adapter + active_step_conditioning + active_engram

    counts = layer_type_counts(effective)
    kv_cache = (
        context_length
        * counts["attention"]
        * 2
        * config.n_kv_heads
        * config.head_dim
        * kv_bits
        // 8
    )
    recurrent_state = (
        counts["gdn2"]
        * config.n_heads
        * config.head_dim
        * config.head_dim
        * state_bits
        // 8
    )
    # Main dense/conv/MoE parameter applications + attention score/value products + LM head.
    attention_flops = (
        4 * counts["attention"] * context_length * config.n_heads * config.head_dim
    )
    gdn_state_flops = (
        6 * counts["gdn2"] * config.n_heads * config.head_dim * config.head_dim
    )
    logits_flops = 2 * config.vocab_size * d
    decode_flops = 2 * active + attention_flops + gdn_state_flops + logits_flops

    return ModelProfile(
        unique_parameters=total,
        engram_parameters=engram,
        unique_backbone_parameters=total - engram,
        active_parameter_applications_per_token=active,
        effective_depth=len(effective),
        unique_depth=len(unique),
        weight_memory_bytes=(total * weight_bits + 7) // 8,
        kv_cache_bytes=kv_cache,
        recurrent_state_bytes=recurrent_state,
        approximate_decode_flops=decode_flops,
        layer_counts=counts,
    )


def human_count(value: int, *, binary: bool = False) -> str:
    base = 1024.0 if binary else 1000.0
    suffixes = ("", "Ki", "Mi", "Gi", "Ti") if binary else ("", "K", "M", "B", "T")
    number = float(value)
    index = 0
    while abs(number) >= base and index < len(suffixes) - 1:
        number /= base
        index += 1
    return f"{number:.2f} {suffixes[index]}"


def render_profile(profile: ModelProfile) -> str:
    return "\n".join(
        (
            f"Stored parameters:          {human_count(profile.unique_parameters)}",
            f"  backbone:                {human_count(profile.unique_backbone_parameters)}",
            f"  Engram tables/fusion:     {human_count(profile.engram_parameters)}",
            f"Active parameter apps/tok: {human_count(profile.active_parameter_applications_per_token)}",
            f"Depth (unique/effective):   {profile.unique_depth}/{profile.effective_depth}",
            f"Effective layer mix:       {profile.layer_counts}",
            f"Quantized weight memory:   {human_count(profile.weight_memory_bytes, binary=True)}B",
            f"KV cache memory:           {human_count(profile.kv_cache_bytes, binary=True)}B",
            f"Recurrent-state memory:    {human_count(profile.recurrent_state_bytes, binary=True)}B",
            f"Approx. decode compute:    {human_count(profile.approximate_decode_flops)}FLOP/token",
        )
    )

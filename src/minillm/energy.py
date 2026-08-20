"""Transparent memory-traffic energy proxies for architecture comparison.

Absolute values are Fermi estimates, not power measurements. The useful signal is the
breakdown and relative sensitivity to active weight bytes, context, and precision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .analysis import ModelProfile, profile_model
from .config import MiniLLMConfig


@dataclass(frozen=True)
class HardwareEnergyProfile:
    name: str = "phone-lpddr-proxy"
    memory_pj_per_byte: float = 60.0
    mac_pj: float = 0.5

    def validate(self) -> HardwareEnergyProfile:
        if self.memory_pj_per_byte <= 0 or self.mac_pj <= 0:
            raise ValueError("energy prices must be positive")
        return self


@dataclass(frozen=True)
class DecodeEnergyEstimate:
    hardware: str
    context_length: int
    weight_bits: int
    kv_bits: int
    active_weight_bytes: float
    kv_read_bytes: float
    recurrent_state_read_write_bytes: float
    approximate_macs: float
    weight_transport_pj: float
    kv_transport_pj: float
    recurrent_state_transport_pj: float
    compute_pj: float
    total_pj: float

    @property
    def total_mj(self) -> float:
        return self.total_pj / 1e9

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["total_mj"] = self.total_mj
        return result


def estimate_decode_energy(
    config: MiniLLMConfig,
    *,
    context_length: int,
    weight_bits: int = 4,
    kv_bits: int = 8,
    hardware: HardwareEnergyProfile | None = None,
    core_repetitions: int | None = None,
) -> DecodeEnergyEstimate:
    """Estimate batch-1 decode energy from active bytes and arithmetic.

    Assumptions: each active matrix weight and the full attention KV history are read
    once per generated token; recurrent state is read and written once. All use the
    selected memory-tier price. Activation traffic, tokenizer, sampling, cache
    residency, and kernel inefficiency are omitted.
    """

    if context_length < 1 or weight_bits < 1 or kv_bits < 1:
        raise ValueError("context length and precisions must be positive")
    hardware = (hardware or HardwareEnergyProfile()).validate()
    profile: ModelProfile = profile_model(
        config,
        context_length=context_length,
        weight_bits=weight_bits,
        kv_bits=kv_bits,
        core_repetitions=core_repetitions,
    )
    # profile.active... excludes the output projection because it is reported as
    # separate logits compute. Include it in weight transport even when tied.
    lm_head_parameters = config.vocab_size * config.d_model
    active_weight_parameters = (
        profile.active_parameter_applications_per_token + lm_head_parameters
    )
    active_weight_bytes = active_weight_parameters * weight_bits / 8
    kv_read_bytes = float(profile.kv_cache_bytes)
    # Recurrent state is both consumed and replaced for each generated token.
    state_read_write_bytes = float(2 * profile.recurrent_state_bytes)
    approximate_macs = profile.approximate_decode_flops / 2
    weight_energy = active_weight_bytes * hardware.memory_pj_per_byte
    kv_energy = kv_read_bytes * hardware.memory_pj_per_byte
    state_energy = state_read_write_bytes * hardware.memory_pj_per_byte
    compute_energy = approximate_macs * hardware.mac_pj
    return DecodeEnergyEstimate(
        hardware=hardware.name,
        context_length=context_length,
        weight_bits=weight_bits,
        kv_bits=kv_bits,
        active_weight_bytes=active_weight_bytes,
        kv_read_bytes=kv_read_bytes,
        recurrent_state_read_write_bytes=state_read_write_bytes,
        approximate_macs=approximate_macs,
        weight_transport_pj=weight_energy,
        kv_transport_pj=kv_energy,
        recurrent_state_transport_pj=state_energy,
        compute_pj=compute_energy,
        total_pj=weight_energy + kv_energy + state_energy + compute_energy,
    )

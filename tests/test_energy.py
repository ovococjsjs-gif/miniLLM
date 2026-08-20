from minillm.config import MiniLLMConfig
from minillm.energy import HardwareEnergyProfile, estimate_decode_energy


def test_energy_proxy_responds_to_precision_and_context() -> None:
    config = MiniLLMConfig(
        vocab_size=128,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        ffn_hidden=64,
        prelude_layers=(),
        core_layers=("attention", "conv"),
        coda_layers=(),
        core_repetitions=1,
        max_core_repetitions=1,
        recurrent_input_injection=False,
        mtp_depth=0,
    ).validate()
    short = estimate_decode_energy(config, context_length=32, weight_bits=4)
    long = estimate_decode_energy(config, context_length=256, weight_bits=4)
    wide_weights = estimate_decode_energy(config, context_length=32, weight_bits=8)
    assert long.kv_transport_pj > short.kv_transport_pj
    assert wide_weights.weight_transport_pj == 2 * short.weight_transport_pj
    assert short.total_mj > 0


def test_recurrent_state_accounts_for_read_and_write() -> None:
    config = MiniLLMConfig(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        head_dim=8,
        ffn_hidden=32,
        prelude_layers=(),
        core_layers=("gdn2",),
        coda_layers=(),
        core_repetitions=1,
        max_core_repetitions=1,
        recurrent_input_injection=False,
        mtp_depth=0,
    ).validate()
    estimate = estimate_decode_energy(config, context_length=32)
    assert estimate.kv_read_bytes == 0
    assert estimate.recurrent_state_read_write_bytes > 0
    assert estimate.recurrent_state_transport_pj > 0


def test_hardware_memory_price_scales_transport_only() -> None:
    config = MiniLLMConfig().validate()
    cheap = estimate_decode_energy(
        config,
        context_length=32,
        hardware=HardwareEnergyProfile("cheap", memory_pj_per_byte=10, mac_pj=1),
    )
    costly = estimate_decode_energy(
        config,
        context_length=32,
        hardware=HardwareEnergyProfile("costly", memory_pj_per_byte=50, mac_pj=1),
    )
    assert costly.compute_pj == cheap.compute_pj
    assert costly.weight_transport_pj == 5 * cheap.weight_transport_pj

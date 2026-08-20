from pathlib import Path

import torch

from minillm.analysis import profile_model
from minillm.config import MiniLLMConfig, MoEConfig
from minillm.model import MiniLLM

ROOT = Path(__file__).parents[1]


def test_full_model_loss_and_parameter_accounting() -> None:
    torch.manual_seed(3)
    config = MiniLLMConfig.load(ROOT / "configs" / "toy.json")
    model = MiniLLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 12))
    output = model(ids, labels=ids)
    assert output.logits.shape == (2, 12, config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    exact = sum(parameter.numel() for parameter in model.parameters())
    estimated = profile_model(config).unique_parameters
    assert exact == estimated


def test_causal_prefix_is_unchanged_by_future_tokens() -> None:
    torch.manual_seed(4)
    config = MiniLLMConfig.load(ROOT / "configs" / "toy.json")
    model = MiniLLM(config).eval()
    first = torch.randint(0, config.vocab_size, (1, 10))
    second = first.clone()
    second[:, 6:] = torch.randint(0, config.vocab_size, (1, 4))
    with torch.no_grad():
        a = model(first).logits
        b = model(second).logits
    torch.testing.assert_close(a[:, :6], b[:, :6], rtol=1e-4, atol=1e-5)


def test_more_recurrences_do_not_change_parameter_count() -> None:
    config = MiniLLMConfig.load(ROOT / "configs" / "toy.json")
    model = MiniLLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 8))
    params_before = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        shallow = model(ids, core_repetitions=1).logits
        deep = model(ids, core_repetitions=4).logits
    params_after = sum(parameter.numel() for parameter in model.parameters())
    assert params_before == params_after
    assert not torch.allclose(shallow, deep)


def test_gradient_checkpointing_preserves_loss_and_gradients() -> None:
    torch.manual_seed(12)
    config = MiniLLMConfig(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        head_dim=8,
        ffn_hidden=32,
        prelude_layers=(),
        core_layers=("attention", "conv"),
        coda_layers=(),
        core_repetitions=1,
        max_core_repetitions=1,
        recurrent_input_injection=False,
        mtp_depth=0,
    ).validate()
    baseline = MiniLLM(config).train()
    checkpointed = MiniLLM(config).train()
    checkpointed.load_state_dict(baseline.state_dict())
    checkpointed.set_gradient_checkpointing(True)
    token_ids = torch.randint(0, config.vocab_size, (2, 8))
    baseline_output = baseline(token_ids, labels=token_ids)
    checkpointed_output = checkpointed(token_ids, labels=token_ids)
    assert baseline_output.loss is not None and checkpointed_output.loss is not None
    torch.testing.assert_close(checkpointed_output.loss, baseline_output.loss)
    checkpointed_output.loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in checkpointed.parameters()
    )


def test_moe_parameter_accounting_matches_model() -> None:
    config = MiniLLMConfig(
        vocab_size=80,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        ffn_hidden=64,
        prelude_layers=(),
        core_layers=("conv", "attention"),
        coda_layers=(),
        core_repetitions=1,
        max_core_repetitions=1,
        recurrent_input_injection=False,
        mtp_depth=0,
        moe=MoEConfig(
            enabled=True,
            num_experts=6,
            top_k=2,
            expert_hidden=24,
            shared_expert_hidden=48,
        ),
    ).validate()
    model = MiniLLM(config)
    exact = sum(parameter.numel() for parameter in model.parameters())
    assert exact == profile_model(config).unique_parameters

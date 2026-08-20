from dataclasses import replace

import torch

from minillm.adaptive_depth import masked_depth_consistency_kl
from minillm.analysis import profile_model
from minillm.config import MiniLLMConfig
from minillm.model import MiniLLM


def recurrent_config(*, conditioned: bool) -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=48,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        head_dim=8,
        ffn_hidden=32,
        max_seq_len=16,
        prelude_layers=("attention",),
        core_layers=("conv", "attention"),
        coda_layers=(),
        core_repetitions=2,
        max_core_repetitions=4,
        recurrent_input_injection=True,
        recurrent_step_conditioning=conditioned,
        mtp_depth=0,
    ).validate()


def test_step_conditioning_is_counted_and_receives_gradients() -> None:
    torch.manual_seed(3)
    plain_config = recurrent_config(conditioned=False)
    config = replace(plain_config, recurrent_step_conditioning=True).validate()
    plain_profile = profile_model(plain_config)
    profile = profile_model(config)
    assert profile.unique_parameters - plain_profile.unique_parameters == 4 * 16

    model = MiniLLM(config).train()
    token_ids = torch.randint(0, config.vocab_size, (2, 8))
    output = model(token_ids, labels=token_ids, core_repetitions=3)
    assert output.loss is not None
    output.loss.backward()
    assert model.recurrent_step_embedding is not None
    gradient = model.recurrent_step_embedding.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient[:3].abs().sum() > 0
    assert gradient[3].abs().sum() == 0


def test_step_conditioned_cached_decode_matches_full_forward() -> None:
    torch.manual_seed(4)
    model = MiniLLM(recurrent_config(conditioned=True)).eval()
    token_ids = torch.randint(0, model.config.vocab_size, (2, 7))
    with torch.no_grad():
        expected = model(token_ids, core_repetitions=3).logits
        cache = None
        pieces = []
        for position in range(token_ids.shape[1]):
            output, cache = model.forward_cached(
                token_ids[:, position : position + 1],
                cache,
                core_repetitions=3,
            )
            pieces.append(output.logits)
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, atol=2e-5, rtol=2e-5)


def test_depth_consistency_is_zero_for_equal_logits_and_detaches_teacher() -> None:
    torch.manual_seed(5)
    student = torch.randn(2, 5, 11, requires_grad=True)
    teacher = student.detach().clone().requires_grad_(True)
    labels = torch.tensor(
        [[-100, 1, -100, 3, 4], [-100, 5, 6, -100, 8]], dtype=torch.long
    )
    equal = masked_depth_consistency_kl(student, teacher, labels)
    torch.testing.assert_close(equal, torch.zeros_like(equal), atol=1e-6, rtol=0)

    shifted_student = student + torch.linspace(0, 1, 11)
    loss = masked_depth_consistency_kl(shifted_student, teacher, labels)
    assert loss > 0
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None

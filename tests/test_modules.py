import torch

from minillm.config import EngramConfig, MoEConfig
from minillm.modules.engram import HashedNgramMemory
from minillm.modules.ffn import SparseMoE
from minillm.modules.gdn2 import ReferenceGatedDeltaNet2


def test_engram_is_deterministic_and_causal() -> None:
    torch.manual_seed(0)
    config = EngramConfig(enabled=True, table_size=101, embedding_dim=4)
    module = HashedNgramMemory(16, config)
    hidden = torch.randn(2, 7, 16)
    ids = torch.randint(0, 50, (2, 7))
    first = module(hidden, ids)
    second = module(hidden, ids)
    torch.testing.assert_close(first, second)

    changed = ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 50
    changed_output = module(hidden, changed)
    torch.testing.assert_close(first[:, :-1], changed_output[:, :-1])


def test_sparse_moe_routes_only_top_k_and_backpropagates() -> None:
    torch.manual_seed(1)
    config = MoEConfig(
        enabled=True, num_experts=4, top_k=2, expert_hidden=12, shared_expert_hidden=8
    )
    module = SparseMoE(16, config)
    x = torch.randn(2, 5, 16, requires_grad=True)
    out, aux, stats = module(x)
    assert out.shape == x.shape
    assert int(stats.counts.sum()) == 2 * 5 * config.top_k
    (out.square().mean() + aux).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.router.weight.grad is not None


def test_gdn2_state_is_fixed_size_and_gradients_are_finite() -> None:
    torch.manual_seed(2)
    module = ReferenceGatedDeltaNet2(d_model=16, n_heads=2, head_dim=8)
    x = torch.randn(2, 6, 16, requires_grad=True)
    out, state = module(x, return_state=True)
    assert out.shape == x.shape
    assert state.shape == (2, 2, 8, 8)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

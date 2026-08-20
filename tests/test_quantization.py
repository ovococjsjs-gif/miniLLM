import torch
from torch import nn

from minillm.quantization import (
    QATConfig,
    QATLinear,
    fake_quantize_groupwise,
    prepare_qat,
)


def test_groupwise_fake_quantization_has_straight_through_gradients() -> None:
    torch.manual_seed(13)
    tensor = torch.randn(3, 35, requires_grad=True)
    quantized = fake_quantize_groupwise(tensor, bits=4, group_size=16)
    assert quantized.shape == tensor.shape
    assert not torch.equal(quantized, tensor)
    quantized.sum().backward()
    torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))


def test_prepare_qat_preserves_tied_parameters_and_router_skip() -> None:
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(20, 8)
            self.projection = nn.Linear(8, 8, bias=False)
            self.lm_head = nn.Linear(8, 20, bias=False)
            self.lm_head.weight = self.embedding.weight
            self.router = nn.Linear(8, 3, bias=False)

    model = Tiny()
    prepare_qat(model, QATConfig(group_size=4), skip=("router",))
    assert isinstance(model.projection, QATLinear)
    assert isinstance(model.lm_head, QATLinear)
    assert isinstance(model.router, nn.Linear) and not isinstance(
        model.router, QATLinear
    )
    assert model.lm_head.weight is model.embedding.weight
    output = model.lm_head(model.projection(torch.randn(2, 8)))
    output.sum().backward()
    assert model.projection.weight.grad is not None

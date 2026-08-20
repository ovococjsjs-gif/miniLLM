import torch

from minillm.benchmarks.associative_recall import generate_recall_batch


def test_mqar_generator_targets_only_answers_and_is_reproducible() -> None:
    a = generate_recall_batch(3, generator=torch.Generator().manual_seed(10))
    b = generate_recall_batch(3, generator=torch.Generator().manual_seed(10))
    torch.testing.assert_close(a.input_ids, b.input_ids)
    torch.testing.assert_close(a.answers, b.answers)
    assert (a.labels != -100).sum().item() == 3 * 2
    batch = torch.arange(3)[:, None]
    torch.testing.assert_close(
        a.input_ids[batch, a.prediction_positions + 1], a.answers
    )

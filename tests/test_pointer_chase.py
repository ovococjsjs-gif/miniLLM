import torch

from minillm.benchmarks.pointer_chase import generate_pointer_chase_batch


def test_pointer_chase_answers_match_encoded_permutations() -> None:
    batch = generate_pointer_chase_batch(
        32,
        node_count=8,
        fixed_hops=4,
        generator=torch.Generator().manual_seed(9),
    )
    assert batch.input_ids.shape == (32, 22)
    assert torch.all(batch.hops == 4)
    assert torch.equal(
        batch.input_ids[
            torch.arange(32),
            batch.prediction_positions + 1,
        ],
        batch.answers,
    )
    for row, answer in zip(batch.input_ids, batch.answers):
        mapping_tokens = row[1:17].view(8, 2)
        mapping = {
            int(source) - 4: int(target) - 4 for source, target in mapping_tokens
        }
        current = int(row[-3]) - 4
        for _ in range(4):
            current = mapping[current]
        assert current + 4 == int(answer)

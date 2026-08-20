import numpy as np

from minillm.draft import (
    NgramDraftShelf,
    NgramShelfConfig,
    verify_greedy_candidate,
    wilson_lower_bound,
)


def test_wilson_bound_is_conservative_and_increases_with_support() -> None:
    assert wilson_lower_bound(4, 4) < wilson_lower_bound(40, 40) < 1
    assert wilson_lower_bound(7, 10) < 0.7


def test_shelf_requires_support_and_falls_back_across_orders() -> None:
    shelf = NgramDraftShelf(
        NgramShelfConfig(
            orders=(2, 3),
            minimum_support=4,
            confidence_threshold=0.5,
            confidence_z=1.0,
        )
    )
    for _ in range(8):
        shelf.update([1, 2], 9)
    # A long context has only one observation and is rejected; shorter support wins.
    shelf.update([0, 1, 2], 8)
    candidate = shelf.query([0, 1, 2])
    assert candidate is not None
    assert candidate.token_id == 9
    assert candidate.order == 2
    assert candidate.support == 9


def test_shelf_candidate_is_lossless_when_neural_model_verifies() -> None:
    shelf = NgramDraftShelf(
        NgramShelfConfig(
            orders=(2,),
            minimum_support=4,
            confidence_threshold=0.5,
            confidence_z=1.0,
        )
    )
    for _ in range(10):
        shelf.update([3, 4], 2)
    candidate = shelf.query([3, 4])
    assert candidate is not None
    assert verify_greedy_candidate([0.0, 0.1, 2.0], candidate)
    assert verify_greedy_candidate(np.asarray([0.0, 0.1, 2.0]), candidate)
    assert not verify_greedy_candidate([3.0, 0.1, 2.0], candidate)

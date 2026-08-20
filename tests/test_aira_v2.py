import numpy as np
import pytest
import torch

from minillm.aira import (
    BoundedAssociativeMemory,
    EpisodicFactStore,
    build_compact_shelf,
    evaluate_hierarchical_shelf,
    load_compact_shelf,
    residual_training_weights,
    save_compact_shelf,
)


def test_frozen_hierarchical_shelf_is_accurate_and_reports_bursts() -> None:
    pattern = np.asarray([1, 2, 3, 4, 1, 2, 3, 5], dtype=np.uint32)
    train = np.tile(pattern, 200)
    validation = np.tile(pattern, 20)
    levels = [build_compact_shelf(train, order) for order in (2, 4)]
    result = evaluate_hierarchical_shelf(
        levels,
        validation,
        minimum_support=5,
        confidence_threshold=0.8,
        confidence_z=0.0,
        selection="longest",
    )
    assert result.coverage > 0.9
    assert result.accuracy == 1.0
    assert result.maximum_correct_burst > 10
    assert result.coverage_by_order["4"] > 0


def test_compact_shelf_archive_round_trip(tmp_path) -> None:
    train = np.tile(np.asarray([1, 2, 3, 4], dtype=np.uint32), 20)
    levels = [build_compact_shelf(train, order) for order in (2, 3)]
    archive = tmp_path / "shelf.npz"

    save_compact_shelf(
        archive,
        levels,
        tokenizer_sha256="a" * 64,
        representation="utf8-byte",
    )
    restored = load_compact_shelf(
        archive,
        expected_tokenizer_sha256="a" * 64,
        expected_representation="utf8-byte",
    )

    with pytest.raises(ValueError, match="tokenizers differ"):
        load_compact_shelf(archive, expected_tokenizer_sha256="b" * 64)
    with pytest.raises(ValueError, match="wrong representation"):
        load_compact_shelf(archive, expected_representation="token-ids")
    assert [level.order for level in restored] == [2, 3]
    for expected, actual in zip(levels, restored):
        np.testing.assert_array_equal(actual.context_hashes, expected.context_hashes)
        np.testing.assert_array_equal(actual.top_tokens, expected.top_tokens)
        np.testing.assert_array_equal(actual.totals, expected.totals)
        np.testing.assert_array_equal(actual.top_counts, expected.top_counts)


def test_wilson_gate_rejects_under_supported_contexts() -> None:
    train = np.asarray([1, 2, 3, 1, 2, 3, 1, 2, 4], dtype=np.uint32)
    validation = np.asarray([1, 2, 3, 1, 2, 3], dtype=np.uint32)
    level = build_compact_shelf(train, 2)
    permissive = evaluate_hierarchical_shelf(
        [level],
        validation,
        minimum_support=1,
        confidence_threshold=0.5,
        confidence_z=0.0,
    )
    conservative = evaluate_hierarchical_shelf(
        [level],
        validation,
        minimum_support=1,
        confidence_threshold=0.5,
        confidence_z=1.96,
    )
    assert conservative.coverage < permissive.coverage


def test_residual_weights_keep_anchor_and_focus_surprise() -> None:
    probabilities = torch.tensor([1.0, 0.75, 0.0])
    weights = residual_training_weights(probabilities, floor=0.1, exponent=2)
    torch.testing.assert_close(weights, torch.tensor([0.1, 0.15625, 1.0]))
    with pytest.raises(ValueError, match="probabilities"):
        residual_training_weights(torch.tensor([1.1]))


def test_bounded_associative_memory_one_shot_gate_and_overwrite() -> None:
    memory = BoundedAssociativeMemory(
        capacity=2,
        dimension=16,
        similarity_threshold=0.5,
        margin_threshold=0.1,
    )
    first = np.r_[np.ones(8), -np.ones(8)]
    second = -first
    third = np.r_[np.ones(4), -np.ones(12)]
    memory.write(first, "first", provenance={"turn": 1})
    memory.write(second, "second", provenance={"turn": 2})
    hit = memory.query(first)
    assert hit.accepted and hit.payload == "first"
    assert hit.provenance == {"turn": 1}
    assert memory.scan_operations() == 32

    memory.write(third, "third", provenance={"turn": 3})
    assert memory.size == 2
    assert memory.query(third).payload == "third"
    assert not memory.query(np.ones(16)).accepted


def test_structured_fact_store_normalizes_keys_and_preserves_provenance() -> None:
    store = EpisodicFactStore(capacity=4, dimension=64)
    slot = store.remember(
        "User.Favorite-Color",
        "green",
        provenance={"turn": 7, "editable": True},
    )

    hit = store.recall("USER favorite color")

    assert hit.accepted and hit.payload == "green"
    assert hit.provenance == {"turn": 7, "editable": True}
    assert not store.recall("user.favorite.food").accepted
    assert store.delete(slot)
    assert not store.recall("user.favorite.color").accepted


def test_associative_memory_rejects_conflicting_duplicate_cues() -> None:
    memory = BoundedAssociativeMemory(capacity=3, dimension=16, margin_threshold=0.1)
    cue = np.r_[np.ones(8), -np.ones(8)]
    memory.write(cue, "old", provenance={"source": "a"})
    memory.write(cue, "contradiction", provenance={"source": "b"})

    hit = memory.query(cue)

    assert not hit.accepted
    assert hit.payload is None
    assert hit.margin == pytest.approx(0.0)


def test_associative_memory_deletion_reuses_inactive_slot() -> None:
    memory = BoundedAssociativeMemory(capacity=3, dimension=16)
    first = np.r_[np.ones(8), -np.ones(8)]
    second = -first
    first_slot = memory.write(first, "first")
    memory.write(second, "second")
    memory.write(np.ones(16), "third")

    assert memory.delete(first_slot)
    assert not memory.delete(first_slot)
    assert memory.size == 2
    assert not memory.query(first).accepted

    replacement_slot = memory.write(first, "replacement")
    assert replacement_slot == first_slot
    assert memory.query(first).payload == "replacement"
    assert memory.size == 3

from __future__ import annotations

import numpy as np
import pytest

from minillm.aira import calibrate_reliability_threshold


def test_calibrator_selects_largest_safe_tied_prefix() -> None:
    scores = np.array([0.99] * 100 + [0.8] * 100 + [0.4] * 100)
    correct = np.array(
        [True] * 100 + [True] * 95 + [False] * 5 + [True] * 20 + [False] * 80
    )

    result = calibrate_reliability_threshold(
        scores,
        correct,
        target_precision=0.9,
        confidence_z=0,
        minimum_accepted=50,
    )

    assert result.threshold == pytest.approx(0.8)
    assert result.accepted_examples == 200
    assert result.empirical_precision == pytest.approx(0.975)
    assert result.accept(scores).sum() == 200


def test_calibrator_rejects_everything_when_precision_is_unproved() -> None:
    result = calibrate_reliability_threshold(
        np.array([0.9, 0.8, 0.7]),
        np.array([True, False, True]),
        target_precision=0.99,
        minimum_accepted=2,
    )

    assert result.threshold is None
    assert not result.accept(np.array([1.0, 0.5])).any()


def test_calibration_validates_shapes() -> None:
    with pytest.raises(ValueError, match="equal one-dimensional"):
        calibrate_reliability_threshold(np.ones(3), np.ones(2, dtype=bool))

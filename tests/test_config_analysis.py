from pathlib import Path

import pytest

from minillm.analysis import profile_model
from minillm.config import MiniLLMConfig

ROOT = Path(__file__).parents[1]


def test_all_configs_validate_and_profile() -> None:
    for path in (ROOT / "configs").glob("*.json"):
        config = MiniLLMConfig.load(path)
        profile = profile_model(config, context_length=1024)
        assert profile.unique_parameters > 0
        assert profile.active_parameter_applications_per_token > 0
        assert (
            profile.effective_depth >= profile.unique_depth
            or config.core_repetitions == 1
        )
        assert profile.weight_memory_bytes > 0


def test_bad_head_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="d_model"):
        MiniLLMConfig(d_model=63).validate()


def test_recurrence_changes_compute_not_stored_weights() -> None:
    config = MiniLLMConfig.load(ROOT / "configs" / "edge_recursive_200m.json")
    shallow = profile_model(config, core_repetitions=1)
    deep = profile_model(config, core_repetitions=6)
    assert shallow.unique_parameters == deep.unique_parameters
    assert shallow.weight_memory_bytes == deep.weight_memory_bytes
    assert (
        deep.active_parameter_applications_per_token
        > shallow.active_parameter_applications_per_token
    )

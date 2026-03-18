import numpy as np
import pytest

from convert_dataset_add_tcp_action import (
    REQUIRED_ACTION_NAMES,
    TCP_ACTION_NAMES,
    _build_new_action,
    _build_output_action_names,
)


def test_build_output_action_names_appends_tcp():
    action_names = list(REQUIRED_ACTION_NAMES)
    output_action_names = _build_output_action_names(action_names, overwrite_tcp=False)
    assert output_action_names == [*action_names, *TCP_ACTION_NAMES]


def test_build_output_action_names_rejects_existing_tcp_without_overwrite():
    action_names = [*REQUIRED_ACTION_NAMES, *TCP_ACTION_NAMES]
    with pytest.raises(ValueError, match="already contains tcp fields"):
        _build_output_action_names(action_names, overwrite_tcp=False)


def test_build_new_action_appends_tcp(monkeypatch):
    monkeypatch.setattr(
        "convert_dataset_add_tcp_action.kinematics_forward",
        lambda joints: {"x": 1.0, "y": 2.0, "z": 3.0, "rx": 4.0, "ry": 5.0, "rz": 6.0},
    )
    action_names = list(REQUIRED_ACTION_NAMES)
    output_action_names = [*action_names, *TCP_ACTION_NAMES]
    action_values = np.asarray([11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 80.0, 1.0, 0.0], dtype=np.float32)

    new_action = _build_new_action(action_values, action_names, output_action_names)

    np.testing.assert_allclose(new_action, np.asarray([11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 80.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32))

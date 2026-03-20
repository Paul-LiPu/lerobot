from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import lerobot_robot_romoya  # noqa: F401
from prepare_romoya_dataset import _transform_file
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import _resolve_raw_feature_names_from_ds_meta, make_pre_post_processors
from repair_romoya_checkpoint_schema import _patch_processor_config, _patch_top_level_config
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_robot_romoya.policies.configuration_act_romoya import (
    ABSOLUTE_ACTION_MODE,
    ABSOLUTE_TCP_ACTION_MODE,
    ACTRomoyaConfig,
    DELTA_ACTION_MODE,
    DELTA_TCP_ACTION_MODE,
)
from lerobot_robot_romoya.policies.romoya_transforms import (
    DEFAULT_ROMOYA_ACTION_NAMES,
    DEFAULT_ROMOYA_OBS_STATE_NAMES,
    DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES,
    JOINT_ACTION_NAMES,
    TCP_ACTION_NAMES,
)


def _make_config(
    *,
    action_mode: str | None = DELTA_ACTION_MODE,
    raw_action_feature_names: list[str] | None = None,
    raw_state_feature_names: list[str] | None = None,
    state_shape: int | None = None,
    action_shape: int | None = None,
    **kwargs,
) -> ACTRomoyaConfig:
    raw_state_feature_names = raw_state_feature_names or list(DEFAULT_ROMOYA_OBS_STATE_NAMES)
    raw_action_feature_names = raw_action_feature_names or list(DEFAULT_ROMOYA_ACTION_NAMES)
    input_features = {
        OBS_STATE: PolicyFeature(
            type=FeatureType.STATE,
            shape=(state_shape or len(raw_state_feature_names),),
        ),
        "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
    }
    output_features = {
        ACTION: PolicyFeature(
            type=FeatureType.ACTION,
            shape=(action_shape or len(raw_action_feature_names),),
        )
    }
    cfg = ACTRomoyaConfig(
        device="cpu",
        input_features=input_features,
        output_features=output_features,
        action_mode=action_mode,
        raw_observation_state_feature_names=list(raw_state_feature_names),
        raw_action_feature_names=list(raw_action_feature_names),
        **kwargs,
    )
    cfg.validate_features()
    return cfg


def _make_stats(cfg: ACTRomoyaConfig) -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {
            "mean": torch.zeros(len(cfg.state_feature_names)),
            "std": torch.ones(len(cfg.state_feature_names)),
            "min": torch.zeros(len(cfg.state_feature_names)),
            "max": torch.ones(len(cfg.state_feature_names)),
        },
        ACTION: {
            "mean": torch.zeros(len(cfg.action_feature_names)),
            "std": torch.ones(len(cfg.action_feature_names)),
            "min": torch.zeros(len(cfg.action_feature_names)),
            "max": torch.ones(len(cfg.action_feature_names)),
        },
    }


def _sample_raw_state() -> torch.Tensor:
    return torch.tensor(
        [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 24],
        dtype=torch.float32,
    )


def _sample_raw_action() -> torch.Tensor:
    return torch.tensor([[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 90.0, 1.0, 0.0]], dtype=torch.float32)


def test_legacy_delta_mode_populates_generic_fields():
    cfg = _make_config()
    assert cfg.state_feature_names == [*JOINT_ACTION_NAMES, "gripper.pos", "DO_1"]
    assert cfg.action_feature_names == [*JOINT_ACTION_NAMES, "gripper.pos", "DO_1"]
    assert cfg.delta_action == [True, True, True, True, True, True, True, False]
    assert cfg.transformed_action_names == [
        "delta_joint1.pos",
        "delta_joint2.pos",
        "delta_joint3.pos",
        "delta_joint4.pos",
        "delta_joint5.pos",
        "delta_joint6.pos",
        "delta_gripper.pos",
        "DO_1",
    ]


def test_generic_absolute_joint_config_validates():
    cfg = _make_config(
        action_mode=None,
        state_shape=8,
        action_shape=8,
        state_feature_names=[*JOINT_ACTION_NAMES, "gripper.pos", "DO_1"],
        action_feature_names=[*JOINT_ACTION_NAMES, "gripper.pos", "DO_1"],
        binary_state=[None, None, None, None, None, None, None, 0.5],
        binary_action=[None, None, None, None, None, None, None, (0.5, 0.0, 1.0)],
        delta_action=[False, False, False, False, False, False, False, False],
    )
    assert cfg.control_schema == "joint_control"
    assert cfg.output_features[ACTION].shape == (8,)


def test_generic_config_rejects_partial_joint_action_schema():
    with pytest.raises(ValueError, match="full joint control"):
        _make_config(
            action_mode=None,
            state_shape=7,
            action_shape=7,
            state_feature_names=[*JOINT_ACTION_NAMES, "DO_1"],
            action_feature_names=[*JOINT_ACTION_NAMES[:-1], "gripper.pos", "DO_1"],
            binary_state=[None] * 6 + [0.5],
            binary_action=[None] * 6 + [(0.5, 0.0, 1.0)],
            delta_action=[False] * 7,
        )


def test_generic_config_rejects_binary_delta_same_action_dim():
    with pytest.raises(ValueError, match="cannot be both binary and delta"):
        _make_config(
            action_mode=None,
            state_shape=8,
            action_shape=8,
            state_feature_names=[*JOINT_ACTION_NAMES, "gripper.pos", "DO_1"],
            action_feature_names=[*JOINT_ACTION_NAMES, "gripper.pos", "DO_1"],
            binary_state=[None, None, None, None, None, None, None, 0.5],
            binary_action=[None, None, None, None, None, None, None, (0.5, 0.0, 1.0)],
            delta_action=[False, False, False, False, False, False, False, True],
        )


def test_legacy_tcp_mode_requires_full_tcp_action_schema():
    cfg = _make_config(
        action_mode=ABSOLUTE_TCP_ACTION_MODE,
        raw_action_feature_names=list(DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES),
        action_shape=len(DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES),
    )
    assert cfg.control_schema == "tcp_control"
    assert cfg.action_feature_names == [*TCP_ACTION_NAMES, "gripper.pos", "DO_1"]


def test_preprocessor_transforms_raw_live_batch():
    cfg = _make_config()
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=_make_stats(cfg))
    processed = preprocessor(
        {
            OBS_STATE: _sample_raw_state(),
            ACTION: _sample_raw_action(),
            "observation.images.side": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
        }
    )
    torch.testing.assert_close(processed[OBS_STATE][0], torch.tensor([10, 20, 30, 40, 50, 60, 70, 1], dtype=torch.float32))
    torch.testing.assert_close(
        processed[ACTION][0],
        torch.tensor([1, 2, 3, 4, 5, 6, 5, 0], dtype=torch.float32),
    )


def test_preprocessor_leaves_prepared_dataset_batch_unchanged():
    cfg = _make_config(action_mode=ABSOLUTE_ACTION_MODE, state_shape=8, action_shape=8)
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=_make_stats(cfg))
    batch = {
        OBS_STATE: torch.tensor([[1, 2, 3, 4, 5, 6, 7, 1]], dtype=torch.float32),
        ACTION: torch.tensor([[11, 22, 33, 44, 55, 66, 75, 0]], dtype=torch.float32),
        "observation.images.side": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    processed = preprocessor(batch)
    torch.testing.assert_close(processed[OBS_STATE], batch[OBS_STATE])
    torch.testing.assert_close(processed[ACTION], batch[ACTION])


def test_postprocessor_reconstructs_joint_control_from_delta_predictions():
    cfg = _make_config()
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=_make_stats(cfg))
    _ = preprocessor(
        {
            OBS_STATE: _sample_raw_state(),
            "observation.images.side": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
        }
    )
    reconstructed = postprocessor(torch.tensor([[1, 2, 3, 4, 5, 6, 5, 0.6]], dtype=torch.float32))
    torch.testing.assert_close(
        reconstructed[0],
        torch.tensor([11, 22, 33, 44, 55, 66, 75, 80, 1, 1], dtype=torch.float32),
    )


def test_postprocessor_reconstructs_joint_control_from_absolute_predictions():
    cfg = _make_config(action_mode=ABSOLUTE_ACTION_MODE)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=_make_stats(cfg))
    _ = preprocessor(
        {
            OBS_STATE: _sample_raw_state(),
            "observation.images.side": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
        }
    )
    reconstructed = postprocessor(torch.tensor([[11, 22, 33, 44, 55, 66, 75, 0.6]], dtype=torch.float32))
    torch.testing.assert_close(
        reconstructed[0],
        torch.tensor([11, 22, 33, 44, 55, 66, 75, 80, 1, 1], dtype=torch.float32),
    )


def test_postprocessor_reconstructs_tcp_control(monkeypatch):
    monkeypatch.setattr(
        "lerobot_robot_romoya.policies.romoya_transforms.kinematics_inverse",
        lambda tcp_pose, seed_joints: [tcp_pose["x"], tcp_pose["y"], tcp_pose["z"], tcp_pose["rx"], tcp_pose["ry"], tcp_pose["rz"]],
    )
    cfg = _make_config(
        action_mode=DELTA_TCP_ACTION_MODE,
        raw_action_feature_names=list(DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES),
        action_shape=len(DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES),
    )
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=_make_stats(cfg))
    raw_state = torch.tensor(
        [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 18 + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        dtype=torch.float32,
    )
    _ = preprocessor(
        {
            OBS_STATE: raw_state,
            "observation.images.side": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
        }
    )
    reconstructed = postprocessor(torch.tensor([[6, 6, 6, 6, 6, 6, 5, 0.6]], dtype=torch.float32))
    torch.testing.assert_close(
        reconstructed[0],
        torch.tensor([7, 8, 9, 10, 11, 12, 75, 80, 1, 1], dtype=torch.float32),
    )


def test_transform_file_rewrites_state_action_and_stats():
    cfg = _make_config()
    df = pd.DataFrame(
        {
            OBS_STATE: [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 24],
            ACTION: [[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 90.0, 1.0, 0.0]],
            "episode_index": [0],
            "frame_index": [0],
            "timestamp": [0.0],
            "index": [0],
            "task_index": [0],
        }
    )
    transformed_df, stats = _transform_file(df, cfg)
    assert transformed_df[OBS_STATE].iloc[0] == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 1.0]
    assert transformed_df[ACTION].iloc[0] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 0.0]
    assert tuple(stats[OBS_STATE]["mean"].shape) == (8,)
    assert tuple(stats[ACTION]["mean"].shape) == (8,)


def test_factory_restores_raw_schemas_from_prepared_dataset_metadata():
    cfg = _make_config(action_mode=ABSOLUTE_ACTION_MODE, state_shape=7, action_shape=7)
    ds_meta = SimpleNamespace(
        features={
            OBS_STATE: {"names": list(cfg.state_feature_names), "shape": (7,)},
            ACTION: {"names": list(cfg.action_feature_names), "shape": (7,)},
        },
        info={
            "romoya_prepare": {
                "source_raw_observation_state_feature_names": list(DEFAULT_ROMOYA_OBS_STATE_NAMES),
                "source_raw_action_feature_names": list(DEFAULT_ROMOYA_ACTION_NAMES),
            }
        },
    )
    raw_state, raw_action = _resolve_raw_feature_names_from_ds_meta(cfg, ds_meta)
    assert raw_state == list(DEFAULT_ROMOYA_OBS_STATE_NAMES)
    assert raw_action == list(DEFAULT_ROMOYA_ACTION_NAMES)


def test_repair_helpers_patch_saved_romoya_raw_schemas():
    raw_state = list(DEFAULT_ROMOYA_OBS_STATE_NAMES)
    raw_action = list(DEFAULT_ROMOYA_ACTION_NAMES)
    config_data = {
        "raw_observation_state_feature_names": ["joint1.pos"],
        "raw_action_feature_names": ["joint1.pos"],
        "state_feature_names": [*JOINT_ACTION_NAMES, "gripper.pos"],
        "action_feature_names": [*JOINT_ACTION_NAMES, "gripper.pos"],
    }
    processor_data = {
        "steps": [
            {
                "registry_name": "act_romoya_preprocess_v1",
                "config": {
                    "raw_observation_state_feature_names": ["joint1.pos"],
                    "raw_action_feature_names": ["joint1.pos"],
                },
            },
            {
                "registry_name": "act_romoya_postprocess_v1",
                "config": {
                    "raw_observation_state_feature_names": ["joint1.pos"],
                    "raw_action_feature_names": ["joint1.pos"],
                },
            },
        ]
    }
    patched_config = _patch_top_level_config(config_data, raw_state, raw_action)
    patched_processor = _patch_processor_config(processor_data, raw_state, raw_action)
    assert patched_config["raw_observation_state_feature_names"] == raw_state
    assert patched_config["raw_action_feature_names"] == raw_action
    for step in patched_processor["steps"]:
        if step["registry_name"].startswith("act_romoya_"):
            assert step["config"]["raw_observation_state_feature_names"] == raw_state
            assert step["config"]["raw_action_feature_names"] == raw_action

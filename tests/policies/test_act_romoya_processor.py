import pytest
import torch

import lerobot_robot_romoya  # noqa: F401
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_robot_romoya.policies.configuration_act_romoya import (
    ABSOLUTE_ACTION_MODE,
    ABSOLUTE_TCP_ACTION_MODE,
    ACTRomoyaConfig,
    DELTA_ACTION_MODE,
    DELTA_TCP_ACTION_MODE,
    DEFAULT_ROMOYA_TCP_ACTION_NAMES,
    DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES,
    DEFAULT_ROMOYA_OBS_STATE_NAMES,
)


def _make_config(
    action_mode: str = DELTA_ACTION_MODE,
    action_dim: int = 10,
    *,
    state_dim: int = 34,
    raw_action_feature_names: list[str] | None = None,
    validate: bool = True,
) -> ACTRomoyaConfig:
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    cfg_kwargs = dict(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        action_mode=action_mode,
    )
    if raw_action_feature_names is not None:
        cfg_kwargs["raw_action_feature_names"] = list(raw_action_feature_names)
    cfg = ACTRomoyaConfig(**cfg_kwargs)
    if validate:
        cfg.validate_features()
    return cfg


def _make_stats(action_dim: int = 10, state_dim: int = 34):
    return {
        OBS_STATE: {
            "mean": [float(i) for i in range(state_dim)],
            "std": [1.0 for _ in range(state_dim)],
            "min": [float(i - 1) for i in range(state_dim)],
            "max": [float(i + 1) for i in range(state_dim)],
        },
        ACTION: {
            "mean": [float(i) for i in range(action_dim)],
            "std": [1.0 for _ in range(action_dim)],
            "min": [float(i - 1) for i in range(action_dim)],
            "max": [float(i + 1) for i in range(action_dim)],
        },
    }


def test_act_romoya_config_shapes():
    cfg = _make_config()
    assert cfg.input_features[OBS_STATE].shape == (8,)
    assert cfg.output_features[ACTION].shape == (8,)


def test_act_romoya_config_accepts_widened_raw_state_schema():
    cfg = _make_config(state_dim=len(DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES), validate=False)
    cfg.raw_observation_state_feature_names = list(DEFAULT_ROMOYA_OBS_STATE_NAMES)

    cfg.validate_features()

    assert cfg.input_features[OBS_STATE].shape == (8,)


def test_act_romoya_absolute_config_shapes_and_names():
    cfg = _make_config(action_mode=ABSOLUTE_ACTION_MODE)
    assert cfg.input_features[OBS_STATE].shape == (8,)
    assert cfg.output_features[ACTION].shape == (8,)
    assert cfg.transformed_action_names == [
        "joint1.pos",
        "joint2.pos",
        "joint3.pos",
        "joint4.pos",
        "joint5.pos",
        "joint6.pos",
        "gripper.pos",
        "DO_1",
    ]


def test_act_romoya_absolute_config_round_trip(tmp_path):
    cfg = _make_config(action_mode=ABSOLUTE_ACTION_MODE)
    cfg.save_pretrained(tmp_path)

    loaded = ACTRomoyaConfig.from_pretrained(tmp_path)

    assert loaded.action_mode == ABSOLUTE_ACTION_MODE
    assert loaded.input_features[OBS_STATE].shape == (8,)
    assert loaded.output_features[ACTION].shape == (8,)
    assert loaded.transformed_action_names == cfg.transformed_action_names


def test_act_romoya_tcp_config_shapes_and_names():
    cfg = _make_config(action_mode=ABSOLUTE_TCP_ACTION_MODE, action_dim=16)
    assert cfg.input_features[OBS_STATE].shape == (8,)
    assert cfg.output_features[ACTION].shape == (8,)
    assert cfg.transformed_action_names == [
        "tcp.x",
        "tcp.y",
        "tcp.z",
        "tcp.rx",
        "tcp.ry",
        "tcp.rz",
        "gripper.pos",
        "DO_1",
    ]


def test_act_romoya_tcp_config_accepts_new_34d_schema():
    cfg = _make_config(action_mode=ABSOLUTE_TCP_ACTION_MODE, action_dim=34)
    assert cfg.input_features[OBS_STATE].shape == (8,)
    assert cfg.output_features[ACTION].shape == (8,)


def test_act_romoya_joint_modes_accept_richer_raw_action_superset():
    cfg = _make_config(
        action_mode=ABSOLUTE_ACTION_MODE,
        action_dim=len(DEFAULT_ROMOYA_TCP_ACTION_NAMES),
        raw_action_feature_names=DEFAULT_ROMOYA_TCP_ACTION_NAMES,
    )
    assert cfg.output_features[ACTION].shape == (8,)


def test_act_romoya_tcp_modes_accept_richer_raw_action_superset():
    cfg = _make_config(
        action_mode=ABSOLUTE_TCP_ACTION_MODE,
        action_dim=len(DEFAULT_ROMOYA_TCP_ACTION_NAMES),
        raw_action_feature_names=DEFAULT_ROMOYA_TCP_ACTION_NAMES,
    )
    assert cfg.output_features[ACTION].shape == (8,)


def test_act_romoya_tcp_modes_reject_old_10d_action_schema():
    cfg = _make_config(action_mode=ABSOLUTE_TCP_ACTION_MODE, action_dim=10, validate=False)
    with pytest.raises(ValueError, match="Convert the old 10D dataset"):
        cfg.validate_features()


def test_act_romoya_tcp_pretrained_config_accepts_runtime_10d_control_shape():
    cfg = _make_config(action_mode=ABSOLUTE_TCP_ACTION_MODE, action_dim=16, validate=False)
    cfg.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(10,))
    cfg.raw_action_feature_names = [
        "joint1.pos",
        "joint2.pos",
        "joint3.pos",
        "joint4.pos",
        "joint5.pos",
        "joint6.pos",
        "gripper.pos",
        "gripper.force",
        "DO_0",
        "DO_1",
        "tcp.x",
        "tcp.y",
        "tcp.z",
        "tcp.rx",
        "tcp.ry",
        "tcp.rz",
    ]

    cfg.validate_features()

    assert cfg.output_features[ACTION].shape == (8,)
    assert cfg.raw_action_feature_names[-6:] == ["tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz"]


def test_act_romoya_processors_resolve():
    cfg = _make_config()
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=_make_stats())
    assert isinstance(preprocessor, PolicyProcessorPipeline)
    assert isinstance(postprocessor, PolicyProcessorPipeline)


def test_act_romoya_preprocessor_slices_state_and_transforms_action():
    cfg = _make_config()
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=_make_stats())
    batch = {
        OBS_STATE: torch.tensor(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 24],
            dtype=torch.float32,
        ),
        ACTION: torch.tensor([[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 90.0, 1.0, 0.0]], dtype=torch.float32),
        "observation.images.wrist": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    processed = preprocessor(batch)
    assert processed[OBS_STATE].shape[-1] == 8
    torch.testing.assert_close(processed[OBS_STATE][0], torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 1.0]))
    torch.testing.assert_close(
        processed[ACTION][0],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 0.0]),
    )


def test_act_romoya_joint_preprocessor_ignores_extra_raw_action_metadata():
    cfg = _make_config(
        action_mode=ABSOLUTE_ACTION_MODE,
        action_dim=len(DEFAULT_ROMOYA_TCP_ACTION_NAMES),
        raw_action_feature_names=DEFAULT_ROMOYA_TCP_ACTION_NAMES,
    )
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=_make_stats(action_dim=len(DEFAULT_ROMOYA_TCP_ACTION_NAMES)))
    batch = {
        OBS_STATE: torch.tensor(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 24],
            dtype=torch.float32,
        ),
        ACTION: torch.tensor(
            [[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 90.0, 1.0, 0.0] + [0.0] * 24],
            dtype=torch.float32,
        ),
        "observation.images.wrist": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    processed = preprocessor(batch)
    torch.testing.assert_close(
        processed[ACTION][0],
        torch.tensor([11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 0.0]),
    )


def test_act_romoya_preprocessor_ignores_appended_widened_state_fields():
    cfg = _make_config(state_dim=len(DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES), validate=False)
    cfg.raw_observation_state_feature_names = list(DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES)
    cfg.validate_features()
    preprocessor, _ = make_pre_post_processors(
        cfg,
        dataset_stats=_make_stats(state_dim=len(DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES)),
    )
    batch = {
        OBS_STATE: torch.tensor(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 73],
            dtype=torch.float32,
        ),
        ACTION: torch.tensor([[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 90.0, 1.0, 0.0]], dtype=torch.float32),
        "observation.images.wrist": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    processed = preprocessor(batch)
    torch.testing.assert_close(processed[OBS_STATE][0], torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 1.0]))


def test_act_romoya_absolute_preprocessor_slices_state_and_action():
    cfg = _make_config(action_mode=ABSOLUTE_ACTION_MODE)
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=_make_stats())
    batch = {
        OBS_STATE: torch.tensor(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 24],
            dtype=torch.float32,
        ),
        ACTION: torch.tensor([[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 90.0, 1.0, 0.0]], dtype=torch.float32),
        "observation.images.wrist": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    processed = preprocessor(batch)
    torch.testing.assert_close(processed[OBS_STATE][0], torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 1.0]))
    torch.testing.assert_close(
        processed[ACTION][0],
        torch.tensor([11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 0.0]),
    )


def test_act_romoya_absolute_tcp_preprocessor_slices_tcp_action():
    cfg = _make_config(action_mode=ABSOLUTE_TCP_ACTION_MODE, action_dim=16)
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=_make_stats(action_dim=16))
    batch = {
        OBS_STATE: torch.tensor(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 18 + [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]],
            dtype=torch.float32,
        ),
        ACTION: torch.tensor(
            [[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 90.0, 1.0, 0.0, 7.1, 8.2, 9.3, 10.4, 11.5, 12.6]],
            dtype=torch.float32,
        ),
        "observation.images.wrist": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    processed = preprocessor(batch)
    torch.testing.assert_close(
        processed[ACTION][0],
        torch.tensor([7.1, 8.2, 9.3, 10.4, 11.5, 12.6, 75.0, 0.0]),
    )


def test_act_romoya_delta_tcp_preprocessor_computes_deltas():
    cfg = _make_config(action_mode=DELTA_TCP_ACTION_MODE, action_dim=16)
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=_make_stats(action_dim=16))
    batch = {
        OBS_STATE: torch.tensor(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 18 + [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]],
            dtype=torch.float32,
        ),
        ACTION: torch.tensor(
            [[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 90.0, 1.0, 0.0, 7.1, 8.2, 9.3, 10.4, 11.5, 12.6]],
            dtype=torch.float32,
        ),
        "observation.images.wrist": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    processed = preprocessor(batch)
    torch.testing.assert_close(
        processed[ACTION][0],
        torch.tensor([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 5.0, 0.0]),
    )


def test_act_romoya_postprocessor_reconstructs_absolute_action():
    cfg = _make_config()
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=_make_stats())
    observation = {
        OBS_STATE: torch.tensor(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 24],
            dtype=torch.float32,
        ),
        "observation.images.wrist": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    _ = preprocessor(observation)
    action = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 0.6]], dtype=torch.float32)
    reconstructed = postprocessor(action)
    torch.testing.assert_close(
        reconstructed[0],
        torch.tensor([11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 80.0, 1.0, 1.0]),
    )


def test_act_romoya_absolute_postprocessor_passes_through_absolute_action():
    cfg = _make_config(action_mode=ABSOLUTE_ACTION_MODE)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=_make_stats())
    observation = {
        OBS_STATE: torch.tensor(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 24],
            dtype=torch.float32,
        ),
        "observation.images.wrist": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    _ = preprocessor(observation)
    action = torch.tensor([[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 0.6]], dtype=torch.float32)
    reconstructed = postprocessor(action)
    torch.testing.assert_close(
        reconstructed[0],
        torch.tensor([11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 80.0, 1.0, 1.0]),
    )


def test_act_romoya_absolute_tcp_postprocessor_converts_to_joint_targets(monkeypatch):
    monkeypatch.setattr(
        "lerobot_robot_romoya.policies.processor_act_romoya.kinematics_inverse",
        lambda tcp_pose, seed_joints: [tcp_pose["x"], tcp_pose["y"], tcp_pose["z"], tcp_pose["rx"], tcp_pose["ry"], tcp_pose["rz"]],
    )
    cfg = _make_config(action_mode=ABSOLUTE_TCP_ACTION_MODE, action_dim=34)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=_make_stats(action_dim=34))
    observation = {
        OBS_STATE: torch.tensor(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 18 + [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]],
            dtype=torch.float32,
        ),
        "observation.images.wrist": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    _ = preprocessor(observation)
    action = torch.tensor([[7.1, 8.2, 9.3, 10.4, 11.5, 12.6, 75.0, 0.6]], dtype=torch.float32)
    reconstructed = postprocessor(action)
    torch.testing.assert_close(
        reconstructed[0],
        torch.tensor([7.1, 8.2, 9.3, 10.4, 11.5, 12.6, 75.0, 80.0, 1.0, 1.0]),
    )


def test_act_romoya_delta_tcp_postprocessor_reconstructs_absolute_joint_targets(monkeypatch):
    monkeypatch.setattr(
        "lerobot_robot_romoya.policies.processor_act_romoya.kinematics_inverse",
        lambda tcp_pose, seed_joints: [tcp_pose["x"], tcp_pose["y"], tcp_pose["z"], tcp_pose["rx"], tcp_pose["ry"], tcp_pose["rz"]],
    )
    cfg = _make_config(action_mode=DELTA_TCP_ACTION_MODE, action_dim=34)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=_make_stats(action_dim=34))
    observation = {
        OBS_STATE: torch.tensor(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 18 + [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]],
            dtype=torch.float32,
        ),
        "observation.images.wrist": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    _ = preprocessor(observation)
    action = torch.tensor([[6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 5.0, 0.6]], dtype=torch.float32)
    reconstructed = postprocessor(action)
    torch.testing.assert_close(
        reconstructed[0],
        torch.tensor([7.1, 8.2, 9.3, 10.4, 11.5, 12.6, 75.0, 80.0, 1.0, 1.0]),
    )


def test_act_romoya_postprocessor_thresholds_do_without_sigmoid():
    cfg = _make_config()
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=_make_stats())
    observation = {
        OBS_STATE: torch.tensor(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 24],
            dtype=torch.float32,
        ),
        "observation.images.wrist": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    _ = preprocessor(observation)
    action = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]], dtype=torch.float32)
    reconstructed = postprocessor(action)
    torch.testing.assert_close(
        reconstructed[0],
        torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 0.0]),
    )


def test_act_romoya_absolute_postprocessor_thresholds_do_without_sigmoid():
    cfg = _make_config(action_mode=ABSOLUTE_ACTION_MODE)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=_make_stats())
    observation = {
        OBS_STATE: torch.tensor(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 0.0, 1.0] + [0.0] * 24],
            dtype=torch.float32,
        ),
        "observation.images.wrist": torch.zeros(1, 3, 360, 640, dtype=torch.float32),
    }
    _ = preprocessor(observation)
    action = torch.tensor([[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 0.4]], dtype=torch.float32)
    reconstructed = postprocessor(action)
    torch.testing.assert_close(
        reconstructed[0],
        torch.tensor([11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 75.0, 80.0, 0.0, 0.0]),
    )


def test_act_romoya_delta_tcp_config_round_trip(tmp_path):
    cfg = _make_config(action_mode=DELTA_TCP_ACTION_MODE, action_dim=34)
    cfg.save_pretrained(tmp_path)

    loaded = ACTRomoyaConfig.from_pretrained(tmp_path)

    assert loaded.action_mode == DELTA_TCP_ACTION_MODE
    assert loaded.input_features[OBS_STATE].shape == (8,)
    assert loaded.output_features[ACTION].shape == (8,)

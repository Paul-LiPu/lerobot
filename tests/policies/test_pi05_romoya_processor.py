from __future__ import annotations

from types import SimpleNamespace

import pytest

import lerobot_robot_romoya  # noqa: F401
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import _resolve_raw_feature_names_from_ds_meta
from lerobot.processor.tokenizer_processor import TokenizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_robot_romoya.policies.configuration_pi05_romoya import PI05RomoyaConfig
from lerobot_robot_romoya.policies.processor_pi05_romoya import (
    PI05RomoyaPostprocessStep,
    PI05RomoyaPreprocessStep,
    make_pi05_romoya_pre_post_processors,
)
from lerobot_robot_romoya.policies.romoya_transforms import (
    DEFAULT_ROMOYA_ACTION_NAMES,
    DEFAULT_ROMOYA_OBS_STATE_NAMES,
)


def _make_config() -> PI05RomoyaConfig:
    cfg = PI05RomoyaConfig(
        device="cpu",
        max_state_dim=32,
        max_action_dim=32,
        state_feature_names=["joint1.pos", "joint2.pos", "joint3.pos", "joint4.pos", "joint5.pos", "joint6.pos", "gripper.pos"],
        action_feature_names=["joint1.pos", "joint2.pos", "joint3.pos", "joint4.pos", "joint5.pos", "joint6.pos", "gripper.pos"],
        binary_state=[None, None, None, None, None, None, 90.0],
        binary_action=[None, None, None, None, None, None, (90.0, 0.0, 99.0)],
        delta_action=[False, False, False, False, False, False, False],
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
    )
    cfg.validate_features()
    return cfg


def test_pi05_romoya_config_validates_transformed_shapes():
    cfg = _make_config()
    assert cfg.input_features[OBS_STATE].shape == (7,)
    assert cfg.output_features[ACTION].shape == (7,)
    assert cfg.control_schema == "joint_control"


def test_factory_restores_raw_schemas_from_prepared_dataset_metadata_for_pi05():
    cfg = _make_config()
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


def test_pi05_romoya_processors_insert_romoya_steps(monkeypatch):
    monkeypatch.setattr(
        TokenizerProcessorStep,
        "__post_init__",
        lambda self: setattr(self, "input_tokenizer", object()),
    )
    cfg = _make_config()
    preprocessor, postprocessor = make_pi05_romoya_pre_post_processors(config=cfg, dataset_stats=None)
    assert isinstance(preprocessor.steps[1], PI05RomoyaPreprocessStep)
    assert isinstance(postprocessor.steps[1], PI05RomoyaPostprocessStep)

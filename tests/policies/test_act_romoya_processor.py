import torch

import lerobot_robot_romoya  # noqa: F401
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_robot_romoya.policies.configuration_act_romoya import ACTRomoyaConfig


def _make_config() -> ACTRomoyaConfig:
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(34,)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,))}
    cfg = ACTRomoyaConfig(input_features=input_features, output_features=output_features, device="cpu")
    cfg.validate_features()
    return cfg


def _make_stats():
    return {
        OBS_STATE: {
            "mean": [float(i) for i in range(34)],
            "std": [1.0 for _ in range(34)],
            "min": [float(i - 1) for i in range(34)],
            "max": [float(i + 1) for i in range(34)],
        },
        ACTION: {
            "mean": [float(i) for i in range(10)],
            "std": [1.0 for _ in range(10)],
            "min": [float(i - 1) for i in range(10)],
            "max": [float(i + 1) for i in range(10)],
        },
    }


def test_act_romoya_config_shapes():
    cfg = _make_config()
    assert cfg.input_features[OBS_STATE].shape == (8,)
    assert cfg.output_features[ACTION].shape == (8,)


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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, ProcessorStep, ProcessorStepRegistry
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_act_romoya import ACTRomoyaConfig


@dataclass
class _ActRomoyaSharedContext:
    latest_observation_state: Tensor | None = None


_ACT_ROMOYA_CONTEXTS: dict[str, _ActRomoyaSharedContext] = {}


def _get_context(context_id: str) -> _ActRomoyaSharedContext:
    if context_id not in _ACT_ROMOYA_CONTEXTS:
        _ACT_ROMOYA_CONTEXTS[context_id] = _ActRomoyaSharedContext()
    return _ACT_ROMOYA_CONTEXTS[context_id]


def _index_map(names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(names)}


def _to_list_or_scalar(values: Any) -> Any:
    if isinstance(values, Tensor):
        values = values.tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return values


def _slice_stat_vector(stats: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    sliced = {}
    for name, values in stats.items():
        values = _to_list_or_scalar(values)
        if isinstance(values, list) and len(values) > 1:
            sliced[name] = [values[i] for i in indices]
        else:
            sliced[name] = values
    return sliced


def _derive_delta_stats(
    action_stats: dict[str, Any],
    action_index: int,
    observation_stats: dict[str, Any],
    observation_index: int,
) -> dict[str, float]:
    action_mean = float(action_stats["mean"][action_index])
    action_std = float(action_stats["std"][action_index])
    action_min = float(action_stats["min"][action_index])
    action_max = float(action_stats["max"][action_index])
    obs_mean = float(observation_stats["mean"][observation_index])
    obs_std = float(observation_stats["std"][observation_index])
    obs_min = float(observation_stats["min"][observation_index])
    obs_max = float(observation_stats["max"][observation_index])
    return {
        "mean": action_mean - obs_mean,
        "std": (action_std**2 + obs_std**2) ** 0.5,
        "min": action_min - obs_max,
        "max": action_max - obs_min,
    }


def _transform_stats(config: ACTRomoyaConfig, dataset_stats: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]] | None:
    if dataset_stats is None:
        return None

    stats = {
        key: {name: _to_list_or_scalar(values) for name, values in feature_stats.items()}
        for key, feature_stats in dataset_stats.items()
    }
    state_name_to_index = _index_map(config.raw_observation_state_feature_names)
    action_name_to_index = _index_map(config.raw_action_feature_names)

    state_indices = [state_name_to_index[name] for name in config.state_feature_names_to_keep]
    stats[OBS_STATE] = _slice_stat_vector(stats[OBS_STATE], state_indices)

    transformed_action_stats = {"mean": [], "std": [], "min": [], "max": []}
    for name in config.joint_action_names:
        delta_stats = _derive_delta_stats(
            dataset_stats[ACTION],
            action_name_to_index[name],
            dataset_stats[OBS_STATE],
            state_name_to_index[name],
        )
        for stat_name, stat_value in delta_stats.items():
            transformed_action_stats[stat_name].append(stat_value)

    gripper_delta_stats = _derive_delta_stats(
        dataset_stats[ACTION],
        action_name_to_index[config.gripper_action_name],
        dataset_stats[OBS_STATE],
        state_name_to_index[config.gripper_action_name],
    )
    for stat_name, stat_value in gripper_delta_stats.items():
        transformed_action_stats[stat_name].append(stat_value)

    for name in config.do_action_names:
        action_idx = action_name_to_index[name]
        for stat_name in transformed_action_stats:
            transformed_action_stats[stat_name].append(float(dataset_stats[ACTION][stat_name][action_idx]))

    stats[ACTION] = transformed_action_stats
    return stats


@dataclass
@ProcessorStepRegistry.register(name="act_romoya_preprocess_v1")
class ACTRomoyaPreprocessStep(ProcessorStep):
    state_feature_names_to_keep: list[str]
    raw_observation_state_feature_names: list[str]
    raw_action_feature_names: list[str]
    joint_action_names: list[str]
    gripper_action_name: str
    do_action_names: list[str]
    context_id: str = "act_romoya"

    def __post_init__(self) -> None:
        self._state_index = _index_map(self.raw_observation_state_feature_names)
        self._action_index = _index_map(self.raw_action_feature_names)
        self._kept_state_indices = [self._state_index[name] for name in self.state_feature_names_to_keep]
        self._joint_action_indices = [self._action_index[name] for name in self.joint_action_names]
        self._joint_state_indices = [self._state_index[name] for name in self.joint_action_names]
        self._gripper_action_index = self._action_index[self.gripper_action_name]
        self._gripper_state_index = self._state_index[self.gripper_action_name]
        self._do_action_indices = [self._action_index[name] for name in self.do_action_names]
        self._context = _get_context(self.context_id)

    def __call__(self, transition):
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is not None and OBS_STATE in observation:
            raw_state = observation[OBS_STATE]
            self._context.latest_observation_state = (
                raw_state.detach().clone() if isinstance(raw_state, Tensor) else torch.as_tensor(raw_state)
            )
            observation[OBS_STATE] = raw_state[..., self._kept_state_indices]

        action = transition.get(TransitionKey.ACTION)
        if action is not None and self._context.latest_observation_state is not None:
            raw_state = self._context.latest_observation_state.to(device=action.device, dtype=action.dtype)
            if action.ndim == raw_state.ndim + 1:
                raw_state = raw_state.unsqueeze(-2)
            joint_delta = action[..., self._joint_action_indices] - raw_state[..., self._joint_state_indices]
            gripper_delta = action[..., [self._gripper_action_index]] - raw_state[..., [self._gripper_state_index]]
            do_values = action[..., self._do_action_indices]
            transition[TransitionKey.ACTION] = torch.cat([joint_delta, gripper_delta, do_values], dim=-1)

        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "state_feature_names_to_keep": self.state_feature_names_to_keep,
            "raw_observation_state_feature_names": self.raw_observation_state_feature_names,
            "raw_action_feature_names": self.raw_action_feature_names,
            "joint_action_names": self.joint_action_names,
            "gripper_action_name": self.gripper_action_name,
            "do_action_names": self.do_action_names,
            "context_id": self.context_id,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = {ptype: dict(feats) for ptype, feats in features.items()}
        features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(len(self.state_feature_names_to_keep),),
        )
        features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(len(self.joint_action_names) + 1 + len(self.do_action_names),),
        )
        return features


@dataclass
@ProcessorStepRegistry.register(name="act_romoya_postprocess_v1")
class ACTRomoyaPostprocessStep(ProcessorStep):
    raw_observation_state_feature_names: list[str]
    raw_action_feature_names: list[str]
    joint_action_names: list[str]
    gripper_action_name: str
    do_action_names: list[str]
    do_threshold: float
    sigmoid_do_outputs: bool = True
    context_id: str = "act_romoya"
    gripper_min: float = 0.0
    gripper_max: float = 100.0

    def __post_init__(self) -> None:
        self._state_index = _index_map(self.raw_observation_state_feature_names)
        self._action_index = _index_map(self.raw_action_feature_names)
        self._joint_state_indices = [self._state_index[name] for name in self.joint_action_names]
        self._joint_action_indices = [self._action_index[name] for name in self.joint_action_names]
        self._gripper_state_index = self._state_index[self.gripper_action_name]
        self._gripper_action_index = self._action_index[self.gripper_action_name]
        self._gripper_force_state_index = self._state_index["gripper.force"]
        self._gripper_force_action_index = self._action_index["gripper.force"]
        self._do0_action_index = self._action_index["DO_0"]
        self._do1_action_index = self._action_index["DO_1"]
        self._do_action_indices = [self._action_index[name] for name in self.do_action_names]
        self._context = _get_context(self.context_id)

    def __call__(self, transition):
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if self._context.latest_observation_state is None:
            raise ValueError("act_romoya postprocessor requires cached observation.state from the preprocessor.")

        raw_state = self._context.latest_observation_state.to(device=action.device, dtype=action.dtype)
        if action.ndim == raw_state.ndim + 1:
            raw_state = raw_state.unsqueeze(-2)
        reconstructed = torch.zeros(
            *action.shape[:-1], len(self.raw_action_feature_names), device=action.device, dtype=action.dtype
        )

        joint_delta = action[..., : len(self.joint_action_names)]
        gripper_delta = action[..., len(self.joint_action_names) : len(self.joint_action_names) + 1]
        do_values = action[..., len(self.joint_action_names) + 1 :]

        reconstructed[..., self._joint_action_indices] = raw_state[..., self._joint_state_indices] + joint_delta
        reconstructed[..., self._gripper_action_index] = torch.clamp(
            raw_state[..., self._gripper_state_index] + gripper_delta.squeeze(-1),
            min=self.gripper_min,
            max=self.gripper_max,
        )
        reconstructed[..., self._gripper_force_action_index] = raw_state[..., self._gripper_force_state_index]

        if self.sigmoid_do_outputs:
            do_values = torch.sigmoid(do_values)
        do_values = (do_values > self.do_threshold).to(dtype=action.dtype)
        if len(self.do_action_names) == 1 and self.do_action_names[0] == "DO_1":
            reconstructed[..., self._do0_action_index] = do_values[..., 0]
            reconstructed[..., self._do1_action_index] = do_values[..., 0]
        else:
            for column, action_index in enumerate(self._do_action_indices):
                reconstructed[..., action_index] = do_values[..., column]

        transition[TransitionKey.ACTION] = reconstructed
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "raw_observation_state_feature_names": self.raw_observation_state_feature_names,
            "raw_action_feature_names": self.raw_action_feature_names,
            "joint_action_names": self.joint_action_names,
            "gripper_action_name": self.gripper_action_name,
            "do_action_names": self.do_action_names,
            "do_threshold": self.do_threshold,
            "sigmoid_do_outputs": self.sigmoid_do_outputs,
            "context_id": self.context_id,
            "gripper_min": self.gripper_min,
            "gripper_max": self.gripper_max,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = {ptype: dict(feats) for ptype, feats in features.items()}
        features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(len(self.raw_action_feature_names),),
        )
        return features


def make_act_romoya_pre_post_processors(
    config: ACTRomoyaConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    transformed_stats = _transform_stats(config, dataset_stats)
    preprocessor, postprocessor = make_act_pre_post_processors(config=config, dataset_stats=transformed_stats)

    preprocess_step = ACTRomoyaPreprocessStep(
        state_feature_names_to_keep=config.state_feature_names_to_keep,
        raw_observation_state_feature_names=config.raw_observation_state_feature_names,
        raw_action_feature_names=config.raw_action_feature_names,
        joint_action_names=config.joint_action_names,
        gripper_action_name=config.gripper_action_name,
        do_action_names=config.do_action_names,
    )
    postprocess_step = ACTRomoyaPostprocessStep(
        raw_observation_state_feature_names=config.raw_observation_state_feature_names,
        raw_action_feature_names=config.raw_action_feature_names,
        joint_action_names=config.joint_action_names,
        gripper_action_name=config.gripper_action_name,
        do_action_names=config.do_action_names,
        do_threshold=config.do_threshold,
        sigmoid_do_outputs=config.sigmoid_do_outputs,
    )

    preprocessor.steps = [preprocessor.steps[0], preprocessor.steps[1], preprocess_step, *preprocessor.steps[2:]]
    postprocessor.steps = [*postprocessor.steps, postprocess_step]
    preprocessor.name = POLICY_PREPROCESSOR_DEFAULT_NAME
    postprocessor.name = POLICY_POSTPROCESSOR_DEFAULT_NAME
    postprocessor.to_transition = policy_action_to_transition
    postprocessor.to_output = transition_to_policy_action
    return preprocessor, postprocessor

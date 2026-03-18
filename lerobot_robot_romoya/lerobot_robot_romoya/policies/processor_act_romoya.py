from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import ACTION, OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from ..lebai_sdk_utils import kinematics_inverse
from .configuration_act_romoya import (
    ABSOLUTE_ACTION_MODE,
    ABSOLUTE_TCP_ACTION_MODE,
    ACTRomoyaConfig,
    DEFAULT_ROMOYA_ACTION_NAMES,
    DELTA_ACTION_MODE,
    DELTA_TCP_ACTION_MODE,
    TCP_ACTION_NAMES,
)


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


def _is_tcp_mode(action_mode: str) -> bool:
    return action_mode in {ABSOLUTE_TCP_ACTION_MODE, DELTA_TCP_ACTION_MODE}


def _reshape_for_iteration(tensor: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    leading_shape = tuple(tensor.shape[:-1])
    return tensor.reshape(-1, tensor.shape[-1]), leading_shape


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
    if config.action_mode == DELTA_ACTION_MODE:
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
    elif config.action_mode == ABSOLUTE_ACTION_MODE:
        absolute_action_names = [*config.joint_action_names, config.gripper_action_name]
        for name in absolute_action_names:
            action_idx = action_name_to_index[name]
            for stat_name in transformed_action_stats:
                transformed_action_stats[stat_name].append(float(dataset_stats[ACTION][stat_name][action_idx]))
    elif config.action_mode == ABSOLUTE_TCP_ACTION_MODE:
        absolute_action_names = [*TCP_ACTION_NAMES, config.gripper_action_name]
        for name in absolute_action_names:
            action_idx = action_name_to_index[name]
            for stat_name in transformed_action_stats:
                transformed_action_stats[stat_name].append(float(dataset_stats[ACTION][stat_name][action_idx]))
    elif config.action_mode == DELTA_TCP_ACTION_MODE:
        for name in TCP_ACTION_NAMES:
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
    else:
        raise ValueError(f"Unsupported action_mode: {config.action_mode}")

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
    action_mode: str = DELTA_ACTION_MODE
    context_id: str = "act_romoya"

    def __post_init__(self) -> None:
        self._state_index = _index_map(self.raw_observation_state_feature_names)
        self._action_index = _index_map(self.raw_action_feature_names)
        self._kept_state_indices = [self._state_index[name] for name in self.state_feature_names_to_keep]
        self._joint_action_indices = [self._action_index[name] for name in self.joint_action_names]
        self._joint_state_indices = [self._state_index[name] for name in self.joint_action_names]
        self._tcp_action_indices = [self._action_index[name] for name in TCP_ACTION_NAMES if name in self._action_index]
        self._tcp_state_indices = [self._state_index[name] for name in TCP_ACTION_NAMES]
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
            do_values = action[..., self._do_action_indices]
            if self.action_mode == DELTA_ACTION_MODE:
                raw_state = self._context.latest_observation_state.to(device=action.device, dtype=action.dtype)
                if action.ndim == raw_state.ndim + 1:
                    raw_state = raw_state.unsqueeze(-2)
                joint_values = action[..., self._joint_action_indices] - raw_state[..., self._joint_state_indices]
                gripper_values = action[..., [self._gripper_action_index]] - raw_state[..., [self._gripper_state_index]]
            elif self.action_mode == ABSOLUTE_ACTION_MODE:
                joint_values = action[..., self._joint_action_indices]
                gripper_values = action[..., [self._gripper_action_index]]
            elif self.action_mode == ABSOLUTE_TCP_ACTION_MODE:
                joint_values = action[..., self._tcp_action_indices]
                gripper_values = action[..., [self._gripper_action_index]]
            elif self.action_mode == DELTA_TCP_ACTION_MODE:
                raw_state = self._context.latest_observation_state.to(device=action.device, dtype=action.dtype)
                if action.ndim == raw_state.ndim + 1:
                    raw_state = raw_state.unsqueeze(-2)
                joint_values = action[..., self._tcp_action_indices] - raw_state[..., self._tcp_state_indices]
                gripper_values = action[..., [self._gripper_action_index]] - raw_state[..., [self._gripper_state_index]]
            else:
                raise ValueError(f"Unsupported action_mode: {self.action_mode}")
            transition[TransitionKey.ACTION] = torch.cat([joint_values, gripper_values, do_values], dim=-1)

        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "action_mode": self.action_mode,
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
    action_mode: str = DELTA_ACTION_MODE
    sigmoid_do_outputs: bool = True
    context_id: str = "act_romoya"
    gripper_min: float = 0.0
    gripper_max: float = 100.0

    def __post_init__(self) -> None:
        self._state_index = _index_map(self.raw_observation_state_feature_names)
        self._output_action_names = list(DEFAULT_ROMOYA_ACTION_NAMES)
        self._action_index = _index_map(self._output_action_names)
        self._joint_state_indices = [self._state_index[name] for name in self.joint_action_names]
        self._joint_action_indices = [self._action_index[name] for name in self.joint_action_names]
        self._tcp_state_indices = [self._state_index[name] for name in TCP_ACTION_NAMES]
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

        flat_action, action_leading_shape = _reshape_for_iteration(action)
        flat_state, _ = _reshape_for_iteration(raw_state.expand(*action.shape[:-1], raw_state.shape[-1]))
        reconstructed = torch.zeros(
            flat_action.shape[0], len(self._output_action_names), device=action.device, dtype=action.dtype
        )

        joint_dims = len(self.joint_action_names) if not _is_tcp_mode(self.action_mode) else len(TCP_ACTION_NAMES)
        joint_values = flat_action[:, :joint_dims]
        gripper_values = flat_action[:, joint_dims : joint_dims + 1]
        do_values = flat_action[:, joint_dims + 1 :]

        if self.action_mode == DELTA_ACTION_MODE:
            reconstructed[:, self._joint_action_indices] = flat_state[:, self._joint_state_indices] + joint_values
            gripper_target = flat_state[:, self._gripper_state_index] + gripper_values.squeeze(-1)
        elif self.action_mode == ABSOLUTE_ACTION_MODE:
            reconstructed[:, self._joint_action_indices] = joint_values
            gripper_target = gripper_values.squeeze(-1)
        elif self.action_mode in {ABSOLUTE_TCP_ACTION_MODE, DELTA_TCP_ACTION_MODE}:
            seed_joints = flat_state[:, self._joint_state_indices]
            current_tcp = flat_state[:, self._tcp_state_indices]
            if self.action_mode == DELTA_TCP_ACTION_MODE:
                target_tcp = current_tcp + joint_values
                gripper_target = flat_state[:, self._gripper_state_index] + gripper_values.squeeze(-1)
            else:
                target_tcp = joint_values
                gripper_target = gripper_values.squeeze(-1)

            joint_targets = []
            for tcp_row, seed_row in zip(target_tcp, seed_joints, strict=True):
                tcp_pose = {name.split(".", 1)[1]: float(value) for name, value in zip(TCP_ACTION_NAMES, tcp_row, strict=True)}
                joint_targets.append(kinematics_inverse(tcp_pose, [float(v) for v in seed_row]))
            reconstructed[:, self._joint_action_indices] = torch.tensor(
                joint_targets, device=action.device, dtype=action.dtype
            )
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        reconstructed[:, self._gripper_action_index] = torch.clamp(
            gripper_target,
            min=self.gripper_min,
            max=self.gripper_max,
        )
        reconstructed[:, self._gripper_force_action_index] = flat_state[:, self._gripper_force_state_index]

        if self.sigmoid_do_outputs:
            do_values = torch.sigmoid(do_values)
        do_values = (do_values > self.do_threshold).to(dtype=action.dtype)
        if len(self.do_action_names) == 1 and self.do_action_names[0] == "DO_1":
            reconstructed[:, self._do0_action_index] = do_values[:, 0]
            reconstructed[:, self._do1_action_index] = do_values[:, 0]
        else:
            for column, action_index in enumerate(self._do_action_indices):
                reconstructed[:, action_index] = do_values[:, column]

        transition[TransitionKey.ACTION] = reconstructed.reshape(*action_leading_shape, len(self._output_action_names))
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "action_mode": self.action_mode,
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
            shape=(len(self._output_action_names),),
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
        action_mode=config.action_mode,
        state_feature_names_to_keep=config.state_feature_names_to_keep,
        raw_observation_state_feature_names=config.raw_observation_state_feature_names,
        raw_action_feature_names=config.raw_action_feature_names,
        joint_action_names=config.joint_action_names,
        gripper_action_name=config.gripper_action_name,
        do_action_names=config.do_action_names,
    )
    postprocess_step = ACTRomoyaPostprocessStep(
        action_mode=config.action_mode,
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

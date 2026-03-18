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

from .configuration_act_romoya import ACTRomoyaConfig
from .romoya_transforms import (
    DEFAULT_ROMOYA_ACTION_NAMES,
    DEFAULT_ROMOYA_OBS_STATE_NAMES,
    DEFAULT_STATE_FEATURE_NAMES,
    DELTA_ACTION_MODE,
    JOINT_ACTION_NAMES,
    RomoyaTransformSpec,
    legacy_transform_spec,
    reconstruct_robot_action,
    select_and_transform_action,
    select_and_transform_state,
)


@dataclass
class _ActRomoyaSharedContext:
    latest_raw_observation_state: Tensor | None = None


_ACT_ROMOYA_CONTEXTS: dict[str, _ActRomoyaSharedContext] = {}


def _get_context(context_id: str) -> _ActRomoyaSharedContext:
    if context_id not in _ACT_ROMOYA_CONTEXTS:
        _ACT_ROMOYA_CONTEXTS[context_id] = _ActRomoyaSharedContext()
    return _ACT_ROMOYA_CONTEXTS[context_id]


def _resolve_spec(
    *,
    action_mode: str | None,
    state_feature_names: list[str] | None,
    action_feature_names: list[str] | None,
    binary_state: list[float | None] | None,
    binary_action: list[tuple[float, float, float] | None] | None,
    delta_action: list[bool] | None,
    state_feature_names_to_keep: list[str] | None,
    joint_action_names: list[str] | None,
    gripper_action_name: str,
    do_action_names: list[str] | None,
) -> RomoyaTransformSpec:
    if state_feature_names and action_feature_names and binary_state is not None and binary_action is not None and delta_action is not None:
        from .romoya_transforms import classify_control_schema

        return RomoyaTransformSpec(
            state_feature_names=list(state_feature_names),
            action_feature_names=list(action_feature_names),
            binary_state=list(binary_state),
            binary_action=list(binary_action),
            delta_action=list(delta_action),
            control_schema=classify_control_schema(list(action_feature_names)),
        )

    return legacy_transform_spec(
        action_mode or DELTA_ACTION_MODE,
        state_feature_names_to_keep=state_feature_names_to_keep or list(DEFAULT_STATE_FEATURE_NAMES),
        joint_action_names=joint_action_names or list(JOINT_ACTION_NAMES),
        gripper_action_name=gripper_action_name,
        do_action_names=do_action_names or ["DO_1"],
    )


@dataclass
@ProcessorStepRegistry.register(name="act_romoya_preprocess_v1")
class ACTRomoyaPreprocessStep(ProcessorStep):
    raw_observation_state_feature_names: list[str]
    raw_action_feature_names: list[str]
    action_mode: str | None = DELTA_ACTION_MODE
    state_feature_names: list[str] | None = None
    action_feature_names: list[str] | None = None
    binary_state: list[float | None] | None = None
    binary_action: list[tuple[float, float, float] | None] | None = None
    delta_action: list[bool] | None = None
    # Deprecated compatibility fields kept for older saved step configs.
    state_feature_names_to_keep: list[str] | None = None
    joint_action_names: list[str] | None = None
    gripper_action_name: str = "gripper.pos"
    do_action_names: list[str] | None = None
    context_id: str = "act_romoya"

    def __post_init__(self) -> None:
        self._spec = _resolve_spec(
            action_mode=self.action_mode,
            state_feature_names=self.state_feature_names,
            action_feature_names=self.action_feature_names,
            binary_state=self.binary_state,
            binary_action=self.binary_action,
            delta_action=self.delta_action,
            state_feature_names_to_keep=self.state_feature_names_to_keep,
            joint_action_names=self.joint_action_names,
            gripper_action_name=self.gripper_action_name,
            do_action_names=self.do_action_names,
        )
        self._context = _get_context(self.context_id)

    def __call__(self, transition):
        observation = transition.get(TransitionKey.OBSERVATION)
        raw_state = None
        if observation is not None and OBS_STATE in observation:
            maybe_state = observation[OBS_STATE]
            if maybe_state.shape[-1] == len(self.raw_observation_state_feature_names):
                raw_state = maybe_state
                self._context.latest_raw_observation_state = raw_state.detach().clone()
                observation[OBS_STATE] = select_and_transform_state(
                    raw_state,
                    self.raw_observation_state_feature_names,
                    self._spec.state_feature_names,
                    self._spec.binary_state,
                )
            elif maybe_state.shape[-1] != len(self._spec.state_feature_names):
                raise ValueError(
                    "act_romoya preprocessor expected observation.state to already be transformed or match "
                    f"the raw schema. Got trailing dim {maybe_state.shape[-1]}."
                )

        action = transition.get(TransitionKey.ACTION)
        if action is not None and action.shape[-1] == len(self.raw_action_feature_names):
            if raw_state is None:
                cached = self._context.latest_raw_observation_state
                if cached is None:
                    raise ValueError("act_romoya raw action preprocessing requires the matching raw observation.state.")
                raw_state = cached.to(device=action.device, dtype=action.dtype)
            else:
                raw_state = raw_state.to(device=action.device, dtype=action.dtype)

            transition[TransitionKey.ACTION] = select_and_transform_action(
                action,
                raw_state,
                self.raw_action_feature_names,
                self.raw_observation_state_feature_names,
                self._spec.action_feature_names,
                self._spec.binary_action,
                self._spec.delta_action,
            )
        elif action is not None and action.shape[-1] != len(self._spec.action_feature_names):
            raise ValueError(
                "act_romoya preprocessor expected action to already be transformed or match the raw action schema. "
                f"Got trailing dim {action.shape[-1]}."
            )

        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "action_mode": self.action_mode,
            "state_feature_names": self._spec.state_feature_names,
            "action_feature_names": self._spec.action_feature_names,
            "binary_state": self._spec.binary_state,
            "binary_action": self._spec.binary_action,
            "delta_action": self._spec.delta_action,
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
            shape=(len(self._spec.state_feature_names),),
        )
        features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(len(self._spec.action_feature_names),),
        )
        return features


@dataclass
@ProcessorStepRegistry.register(name="act_romoya_postprocess_v1")
class ACTRomoyaPostprocessStep(ProcessorStep):
    raw_observation_state_feature_names: list[str]
    raw_action_feature_names: list[str]
    do_threshold: float
    action_mode: str | None = DELTA_ACTION_MODE
    state_feature_names: list[str] | None = None
    action_feature_names: list[str] | None = None
    binary_state: list[float | None] | None = None
    binary_action: list[tuple[float, float, float] | None] | None = None
    delta_action: list[bool] | None = None
    # Deprecated compatibility fields kept for older saved step configs.
    state_feature_names_to_keep: list[str] | None = None
    joint_action_names: list[str] | None = None
    gripper_action_name: str = "gripper.pos"
    do_action_names: list[str] | None = None
    sigmoid_do_outputs: bool = False
    context_id: str = "act_romoya"
    gripper_min: float = 0.0
    gripper_max: float = 100.0

    def __post_init__(self) -> None:
        self._spec = _resolve_spec(
            action_mode=self.action_mode,
            state_feature_names=self.state_feature_names,
            action_feature_names=self.action_feature_names,
            binary_state=self.binary_state,
            binary_action=self.binary_action,
            delta_action=self.delta_action,
            state_feature_names_to_keep=self.state_feature_names_to_keep,
            joint_action_names=self.joint_action_names,
            gripper_action_name=self.gripper_action_name,
            do_action_names=self.do_action_names,
        )
        self._context = _get_context(self.context_id)

    def __call__(self, transition):
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        raw_state = self._context.latest_raw_observation_state
        if raw_state is None:
            raise ValueError("act_romoya postprocessor requires cached raw observation.state from the preprocessor.")

        raw_state = raw_state.to(device=action.device, dtype=action.dtype)
        if self.sigmoid_do_outputs:
            action = action.clone()
            do_idx = self._spec.action_feature_names.index("DO_1")
            action[..., do_idx] = torch.sigmoid(action[..., do_idx])

        binary_action = []
        for name, spec in zip(self._spec.action_feature_names, self._spec.binary_action, strict=True):
            if spec is None and name == "DO_1":
                binary_action.append((self.do_threshold, 0.0, 1.0))
            else:
                binary_action.append(spec)
        active_spec = RomoyaTransformSpec(
            state_feature_names=self._spec.state_feature_names,
            action_feature_names=self._spec.action_feature_names,
            binary_state=self._spec.binary_state,
            binary_action=binary_action,
            delta_action=self._spec.delta_action,
            control_schema=self._spec.control_schema,
        )
        transition[TransitionKey.ACTION] = reconstruct_robot_action(
            action,
            raw_state,
            active_spec,
            self.raw_observation_state_feature_names,
            gripper_min=self.gripper_min,
            gripper_max=self.gripper_max,
        )
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "action_mode": self.action_mode,
            "state_feature_names": self._spec.state_feature_names,
            "action_feature_names": self._spec.action_feature_names,
            "binary_state": self._spec.binary_state,
            "binary_action": self._spec.binary_action,
            "delta_action": self._spec.delta_action,
            "state_feature_names_to_keep": self.state_feature_names_to_keep,
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
            shape=(len(DEFAULT_ROMOYA_ACTION_NAMES),),
        )
        return features


def make_act_romoya_pre_post_processors(
    config: ACTRomoyaConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    # Dataset prep writes transformed observation.state/action values and their true stats. Training should
    # therefore use the dataset statistics as-is, with no Romoya-specific stat rewriting here.
    preprocessor, postprocessor = make_act_pre_post_processors(config=config, dataset_stats=dataset_stats)

    preprocess_step = ACTRomoyaPreprocessStep(
        action_mode=config.action_mode,
        raw_observation_state_feature_names=config.raw_observation_state_feature_names,
        raw_action_feature_names=config.raw_action_feature_names,
        state_feature_names=config.state_feature_names,
        action_feature_names=config.action_feature_names,
        binary_state=config.binary_state,
        binary_action=config.binary_action,
        delta_action=config.delta_action,
        state_feature_names_to_keep=config.state_feature_names_to_keep,
        joint_action_names=config.joint_action_names,
        gripper_action_name=config.gripper_action_name,
        do_action_names=config.do_action_names,
    )
    postprocess_step = ACTRomoyaPostprocessStep(
        action_mode=config.action_mode,
        raw_observation_state_feature_names=config.raw_observation_state_feature_names,
        raw_action_feature_names=config.raw_action_feature_names,
        state_feature_names=config.state_feature_names,
        action_feature_names=config.action_feature_names,
        binary_state=config.binary_state,
        binary_action=config.binary_action,
        delta_action=config.delta_action,
        state_feature_names_to_keep=config.state_feature_names_to_keep,
        joint_action_names=config.joint_action_names,
        gripper_action_name=config.gripper_action_name,
        do_action_names=config.do_action_names,
        do_threshold=config.do_threshold,
        sigmoid_do_outputs=config.sigmoid_do_outputs,
    )

    preprocessor.add_step(preprocess_step, after="rename_observations_processor")
    postprocessor.add_step(postprocess_step, before="unnormalizer_processor")
    preprocessor.add_step(policy_action_to_transition, before=POLICY_PREPROCESSOR_DEFAULT_NAME)
    preprocessor.add_step(transition_to_policy_action, after=POLICY_PREPROCESSOR_DEFAULT_NAME)
    postprocessor.add_step(policy_action_to_transition, before=POLICY_POSTPROCESSOR_DEFAULT_NAME)
    postprocessor.add_step(transition_to_policy_action, after=POLICY_POSTPROCESSOR_DEFAULT_NAME)
    return preprocessor, postprocessor

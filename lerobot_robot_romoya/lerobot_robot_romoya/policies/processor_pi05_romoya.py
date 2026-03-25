from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, ProcessorStepRegistry
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_pi05_romoya import PI05RomoyaConfig
from .processor_act_romoya import ACTRomoyaPostprocessStep, ACTRomoyaPreprocessStep


@ProcessorStepRegistry.register(name="pi05_romoya_preprocess_v1")
@dataclass
class PI05RomoyaPreprocessStep(ACTRomoyaPreprocessStep):
    pass


@ProcessorStepRegistry.register(name="pi05_romoya_postprocess_v1")
@dataclass
class PI05RomoyaPostprocessStep(ACTRomoyaPostprocessStep):
    pass


def make_pi05_romoya_pre_post_processors(
    config: PI05RomoyaConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)

    preprocess_step = PI05RomoyaPreprocessStep(
        raw_observation_state_feature_names=config.raw_observation_state_feature_names,
        raw_action_feature_names=config.raw_action_feature_names,
        action_mode=None,
        state_feature_names=config.state_feature_names,
        action_feature_names=config.action_feature_names,
        binary_state=config.binary_state,
        binary_action=config.binary_action,
        delta_action=config.delta_action,
    )
    postprocess_step = PI05RomoyaPostprocessStep(
        raw_observation_state_feature_names=config.raw_observation_state_feature_names,
        raw_action_feature_names=config.raw_action_feature_names,
        action_mode=None,
        state_feature_names=config.state_feature_names,
        action_feature_names=config.action_feature_names,
        binary_state=config.binary_state,
        binary_action=config.binary_action,
        delta_action=config.delta_action,
        do_threshold=config.do_threshold,
        gripper_threshold=config.gripper_threshold,
        sigmoid_do_outputs=False,
    )

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[preprocessor.steps[0], preprocess_step, *preprocessor.steps[1:]],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        to_transition=preprocessor.to_transition,
        to_output=preprocessor.to_output,
        before_step_hooks=preprocessor.before_step_hooks,
        after_step_hooks=preprocessor.after_step_hooks,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[postprocessor.steps[0], postprocess_step, *postprocessor.steps[1:]],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=postprocessor.to_transition,
        to_output=postprocessor.to_output,
        before_step_hooks=postprocessor.before_step_hooks,
        after_step_hooks=postprocessor.after_step_hooks,
    )
    return preprocessor, postprocessor


__all__ = ["make_pi05_romoya_pre_post_processors", "PI05RomoyaPreprocessStep", "PI05RomoyaPostprocessStep"]

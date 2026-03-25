from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import ACTION, OBS_STATE

from .romoya_transforms import (
    DEFAULT_ROMOYA_ACTION_NAMES,
    DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES,
    RomoyaTransformSpec,
    validate_transform_spec,
)


@PI05Config.register_subclass("pi05_romoya")
@dataclass
class PI05RomoyaConfig(PI05Config):
    raw_observation_state_feature_names: list[str] = field(
        default_factory=lambda: list(DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES)
    )
    raw_action_feature_names: list[str] = field(default_factory=lambda: list(DEFAULT_ROMOYA_ACTION_NAMES))

    state_feature_names: list[str] = field(default_factory=list)
    action_feature_names: list[str] = field(default_factory=list)
    binary_state: list[float | None] = field(default_factory=list)
    binary_action: list[tuple[float, float, float] | None] = field(default_factory=list)
    delta_action: list[bool] = field(default_factory=list)
    do_threshold: float | None = None
    gripper_threshold: float | None = None

    @property
    def transform_spec(self) -> RomoyaTransformSpec:
        return RomoyaTransformSpec(
            state_feature_names=list(self.state_feature_names),
            action_feature_names=list(self.action_feature_names),
            binary_state=list(self.binary_state),
            binary_action=list(self.binary_action),
            delta_action=list(self.delta_action),
            control_schema=validate_transform_spec(
                self.state_feature_names,
                self.action_feature_names,
                self.binary_state,
                self.binary_action,
                self.delta_action,
                self.raw_observation_state_feature_names,
                self.raw_action_feature_names,
            ),
        )

    @property
    def control_schema(self) -> str:
        return self.transform_spec.control_schema

    def validate_features(self) -> None:
        if OBS_STATE not in (self.input_features or {}):
            raise ValueError("pi05_romoya requires observation.state in input_features.")
        if ACTION not in (self.output_features or {}):
            raise ValueError("pi05_romoya requires action in output_features.")

        spec = self.transform_spec

        raw_state_shape = tuple(self.input_features[OBS_STATE].shape)
        valid_raw_state_shapes = {
            (len(self.raw_observation_state_feature_names),),
            (len(spec.state_feature_names),),
        }
        if raw_state_shape not in valid_raw_state_shapes:
            raise ValueError(
                "Dataset observation.state shape does not match pi05_romoya raw_observation_state_feature_names "
                f"or transformed state schema. Got {raw_state_shape}."
            )

        raw_action_shape = tuple(self.output_features[ACTION].shape)
        valid_raw_action_shapes = {
            (len(self.raw_action_feature_names),),
            (len(spec.action_feature_names),),
        }
        if raw_action_shape not in valid_raw_action_shapes:
            raise ValueError(
                "Dataset action shape does not match pi05_romoya raw_action_feature_names or transformed action "
                f"schema. Got {raw_action_shape}."
            )

        if len(spec.state_feature_names) > self.max_state_dim:
            raise ValueError(
                f"pi05_romoya state dims ({len(spec.state_feature_names)}) exceed max_state_dim ({self.max_state_dim})."
            )
        if len(spec.action_feature_names) > self.max_action_dim:
            raise ValueError(
                f"pi05_romoya action dims ({len(spec.action_feature_names)}) exceed max_action_dim ({self.max_action_dim})."
            )

        self.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(len(spec.state_feature_names),))
        self.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(len(spec.action_feature_names),))
        super().validate_features()


__all__ = ["PI05RomoyaConfig"]

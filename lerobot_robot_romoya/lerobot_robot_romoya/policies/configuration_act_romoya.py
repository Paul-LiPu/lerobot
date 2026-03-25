from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from .romoya_transforms import (
    ABSOLUTE_ACTION_MODE,
    ABSOLUTE_TCP_ACTION_MODE,
    DEFAULT_ROMOYA_ACTION_NAMES,
    DEFAULT_ROMOYA_OBS_STATE_NAMES,
    DEFAULT_ROMOYA_TCP_ACTION_NAMES,
    DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES,
    DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES,
    DEFAULT_STATE_FEATURE_NAMES,
    DELTA_ACTION_MODE,
    DELTA_TCP_ACTION_MODE,
    JOINT_ACTION_NAMES,
    RomoyaTransformSpec,
    TCP_ACTION_NAMES,
    classify_control_schema,
    legacy_transform_spec,
    transformed_action_names,
    validate_transform_spec,
)


@ACTConfig.register_subclass("act_romoya")
@dataclass
class ACTRomoyaConfig(ACTConfig):
    chunk_size: int = 60
    n_action_steps: int = 5
    action_mode: str | None = DELTA_ACTION_MODE

    # Deprecated compatibility fields kept for loading older checkpoints/configs.
    state_feature_names_to_keep: list[str] = field(default_factory=lambda: list(DEFAULT_STATE_FEATURE_NAMES))
    joint_action_names: list[str] = field(default_factory=lambda: list(JOINT_ACTION_NAMES))
    gripper_action_name: str = "gripper.pos"
    do_action_names: list[str] = field(default_factory=lambda: ["DO_1"])

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
    sigmoid_do_outputs: bool = False
    observation_image_resize_shape: list[int] | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def _resolve_transform_spec(self) -> RomoyaTransformSpec:
        if self.action_mode is not None:
            spec = legacy_transform_spec(
                self.action_mode,
                state_feature_names_to_keep=self.state_feature_names_to_keep,
                joint_action_names=self.joint_action_names,
                gripper_action_name=self.gripper_action_name,
                do_action_names=self.do_action_names,
            )
            self.state_feature_names = list(spec.state_feature_names)
            self.action_feature_names = list(spec.action_feature_names)
            self.binary_state = list(spec.binary_state)
            self.binary_action = list(spec.binary_action)
            self.delta_action = list(spec.delta_action)
            return spec

        spec = RomoyaTransformSpec(
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
        return spec

    @property
    def transform_spec(self) -> RomoyaTransformSpec:
        return self._resolve_transform_spec()

    @property
    def transformed_action_names(self) -> list[str]:
        spec = self.transform_spec
        return transformed_action_names(spec.action_feature_names, spec.delta_action)

    @property
    def control_schema(self) -> str:
        return self.transform_spec.control_schema

    def validate_features(self) -> None:
        if OBS_STATE not in (self.input_features or {}):
            raise ValueError("act_romoya requires observation.state in input_features.")
        if ACTION not in (self.output_features or {}):
            raise ValueError("act_romoya requires action in output_features.")

        spec = self._resolve_transform_spec()
        validate_transform_spec(
            spec.state_feature_names,
            spec.action_feature_names,
            spec.binary_state,
            spec.binary_action,
            spec.delta_action,
            self.raw_observation_state_feature_names,
            self.raw_action_feature_names,
        )

        raw_state_shape = tuple(self.input_features[OBS_STATE].shape)
        valid_raw_state_shapes = {
            (len(self.raw_observation_state_feature_names),),
            (len(spec.state_feature_names),),
        }
        if raw_state_shape not in valid_raw_state_shapes:
            raise ValueError(
                "Dataset observation.state shape does not match act_romoya raw_observation_state_feature_names "
                f"or transformed state schema. Got {raw_state_shape}."
            )

        raw_action_shape = tuple(self.output_features[ACTION].shape)
        valid_raw_action_shapes = {
            (len(self.raw_action_feature_names),),
            (len(spec.action_feature_names),),
        }
        if raw_action_shape not in valid_raw_action_shapes:
            raise ValueError(
                "Dataset action shape does not match act_romoya raw_action_feature_names or transformed action "
                f"schema. Got {raw_action_shape}."
            )

        # Training/inference both operate on the transformed Romoya state/action sizes.
        self.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(len(spec.state_feature_names),))
        self.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(len(spec.action_feature_names),))
        super().validate_features()


__all__ = [
    "ABSOLUTE_ACTION_MODE",
    "ABSOLUTE_TCP_ACTION_MODE",
    "ACTRomoyaConfig",
    "DEFAULT_ROMOYA_ACTION_NAMES",
    "DEFAULT_ROMOYA_OBS_STATE_NAMES",
    "DEFAULT_ROMOYA_TCP_ACTION_NAMES",
    "DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES",
    "DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES",
    "DELTA_ACTION_MODE",
    "DELTA_TCP_ACTION_MODE",
    "JOINT_ACTION_NAMES",
    "TCP_ACTION_NAMES",
    "classify_control_schema",
]

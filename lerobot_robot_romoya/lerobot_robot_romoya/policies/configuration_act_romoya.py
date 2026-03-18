from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_STATE


DEFAULT_ROMOYA_OBS_STATE_NAMES = [
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
    "joint1.effort",
    "joint2.effort",
    "joint3.effort",
    "joint4.effort",
    "joint5.effort",
    "joint6.effort",
    "joint1.vel",
    "joint2.vel",
    "joint3.vel",
    "joint4.vel",
    "joint5.vel",
    "joint6.vel",
    "joint1.acc",
    "joint2.acc",
    "joint3.acc",
    "joint4.acc",
    "joint5.acc",
    "joint6.acc",
    "tcp.x",
    "tcp.y",
    "tcp.z",
    "tcp.rx",
    "tcp.ry",
    "tcp.rz",
]

DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES = [
    *DEFAULT_ROMOYA_OBS_STATE_NAMES,
    "joint1.temp",
    "joint2.temp",
    "joint3.temp",
    "joint4.temp",
    "joint5.temp",
    "joint6.temp",
    "joint1.voltage",
    "joint2.voltage",
    "joint3.voltage",
    "joint4.voltage",
    "joint5.voltage",
    "joint6.voltage",
    "flange_voltage",
    "flange.x",
    "flange.y",
    "flange.z",
    "flange.rx",
    "flange.ry",
    "flange.rz",
    "target_joint1.pos",
    "target_joint2.pos",
    "target_joint3.pos",
    "target_joint4.pos",
    "target_joint5.pos",
    "target_joint6.pos",
    "target_joint1.vel",
    "target_joint2.vel",
    "target_joint3.vel",
    "target_joint4.vel",
    "target_joint5.vel",
    "target_joint6.vel",
    "target_joint1.acc",
    "target_joint2.acc",
    "target_joint3.acc",
    "target_joint4.acc",
    "target_joint5.acc",
    "target_joint6.acc",
    "target_joint1.effort",
    "target_joint2.effort",
    "target_joint3.effort",
    "target_joint4.effort",
    "target_joint5.effort",
    "target_joint6.effort",
    "target_tcp.x",
    "target_tcp.y",
    "target_tcp.z",
    "target_tcp.rx",
    "target_tcp.ry",
    "target_tcp.rz",
]

DEFAULT_ROMOYA_ACTION_NAMES = [
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
]

DEFAULT_ROMOYA_TCP_ACTION_NAMES = [
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
    "joint1.effort",
    "joint2.effort",
    "joint3.effort",
    "joint4.effort",
    "joint5.effort",
    "joint6.effort",
    "joint1.vel",
    "joint2.vel",
    "joint3.vel",
    "joint4.vel",
    "joint5.vel",
    "joint6.vel",
    "joint1.acc",
    "joint2.acc",
    "joint3.acc",
    "joint4.acc",
    "joint5.acc",
    "joint6.acc",
    "tcp.x",
    "tcp.y",
    "tcp.z",
    "tcp.rx",
    "tcp.ry",
    "tcp.rz",
]

DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES = [
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

DELTA_ACTION_MODE = "delta_joint_gripper_do1"
ABSOLUTE_ACTION_MODE = "absolute_joint_gripper_do1"
ABSOLUTE_TCP_ACTION_MODE = "absolute_tcp_gripper_do1"
DELTA_TCP_ACTION_MODE = "delta_tcp_gripper_do1"

TCP_ACTION_NAMES = ["tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz"]


@ACTConfig.register_subclass("act_romoya")
@dataclass
class ACTRomoyaConfig(ACTConfig):
    chunk_size: int = 60
    n_action_steps: int = 5
    state_feature_names_to_keep: list[str] = field(
        default_factory=lambda: [
            "joint1.pos",
            "joint2.pos",
            "joint3.pos",
            "joint4.pos",
            "joint5.pos",
            "joint6.pos",
            "gripper.pos",
            "DO_1",
        ]
    )
    action_mode: str = DELTA_ACTION_MODE
    joint_action_names: list[str] = field(
        default_factory=lambda: [
            "joint1.pos",
            "joint2.pos",
            "joint3.pos",
            "joint4.pos",
            "joint5.pos",
            "joint6.pos",
        ]
    )
    gripper_action_name: str = "gripper.pos"
    do_action_names: list[str] = field(default_factory=lambda: ["DO_1"])
    raw_observation_state_feature_names: list[str] = field(
        default_factory=lambda: list(DEFAULT_ROMOYA_OBS_STATE_NAMES)
    )
    raw_action_feature_names: list[str] = field(default_factory=lambda: list(DEFAULT_ROMOYA_ACTION_NAMES))
    do_threshold: float = 0.5
    sigmoid_do_outputs: bool = False

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    @property
    def transformed_action_names(self) -> list[str]:
        if self.action_mode == DELTA_ACTION_MODE:
            return [
                *(f"delta_{name}" for name in self.joint_action_names),
                f"delta_{self.gripper_action_name}",
                *self.do_action_names,
            ]
        if self.action_mode == ABSOLUTE_ACTION_MODE:
            return [
                *self.joint_action_names,
                self.gripper_action_name,
                *self.do_action_names,
            ]
        if self.action_mode == ABSOLUTE_TCP_ACTION_MODE:
            return [*TCP_ACTION_NAMES, self.gripper_action_name, *self.do_action_names]
        if self.action_mode == DELTA_TCP_ACTION_MODE:
            return [
                *(f"delta_{name}" for name in TCP_ACTION_NAMES),
                f"delta_{self.gripper_action_name}",
                *self.do_action_names,
            ]
        raise ValueError(f"Unsupported action_mode: {self.action_mode}")

    @property
    def required_action_names(self) -> list[str]:
        if self.action_mode in {DELTA_ACTION_MODE, ABSOLUTE_ACTION_MODE}:
            return [*self.joint_action_names, self.gripper_action_name, *self.do_action_names]
        if self.action_mode in {ABSOLUTE_TCP_ACTION_MODE, DELTA_TCP_ACTION_MODE}:
            return [*TCP_ACTION_NAMES, self.gripper_action_name, *self.do_action_names]
        raise ValueError(f"Unsupported action_mode: {self.action_mode}")

    def validate_features(self) -> None:
        if OBS_STATE not in (self.input_features or {}):
            raise ValueError("act_romoya requires observation.state in input_features.")
        if ACTION not in (self.output_features or {}):
            raise ValueError("act_romoya requires action in output_features.")
        valid_action_modes = {
            DELTA_ACTION_MODE,
            ABSOLUTE_ACTION_MODE,
            ABSOLUTE_TCP_ACTION_MODE,
            DELTA_TCP_ACTION_MODE,
        }
        if self.action_mode not in valid_action_modes:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        missing_state = set(self.state_feature_names_to_keep) - set(self.raw_observation_state_feature_names)
        if missing_state:
            raise ValueError(f"Unknown state feature names: {sorted(missing_state)}")

        raw_state_feature = self.input_features[OBS_STATE]
        raw_action_feature = self.output_features[ACTION]
        raw_state_dim = len(self.raw_observation_state_feature_names)
        transformed_state_dim = len(self.state_feature_names_to_keep)
        transformed_action_dim = len(self.transformed_action_names)
        is_tcp_mode = self.action_mode in {ABSOLUTE_TCP_ACTION_MODE, DELTA_TCP_ACTION_MODE}

        required_action_names = set(self.required_action_names)
        required_tcp_names = set(TCP_ACTION_NAMES) | {self.gripper_action_name} | set(self.do_action_names)

        required_state_names = set(self.state_feature_names_to_keep)
        if raw_action_feature.shape[0] == len(DEFAULT_ROMOYA_ACTION_NAMES) and not self.raw_action_feature_names:
            self.raw_action_feature_names = list(DEFAULT_ROMOYA_ACTION_NAMES)
        elif raw_action_feature.shape[0] == len(DEFAULT_ROMOYA_ACTION_NAMES) and self.raw_action_feature_names == list(
            DEFAULT_ROMOYA_ACTION_NAMES
        ):
            self.raw_action_feature_names = list(DEFAULT_ROMOYA_ACTION_NAMES)
        elif raw_action_feature.shape[0] == len(DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES) and not self.raw_action_feature_names:
            self.raw_action_feature_names = list(DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES)
        elif self.raw_action_feature_names == list(DEFAULT_ROMOYA_ACTION_NAMES) and raw_action_feature.shape[0] == len(
            DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES
        ):
            self.raw_action_feature_names = list(DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES)
        elif raw_action_feature.shape[0] == len(DEFAULT_ROMOYA_TCP_ACTION_NAMES) and not self.raw_action_feature_names:
            self.raw_action_feature_names = list(DEFAULT_ROMOYA_TCP_ACTION_NAMES)
        elif self.raw_action_feature_names == list(DEFAULT_ROMOYA_ACTION_NAMES) and raw_action_feature.shape[0] == len(
            DEFAULT_ROMOYA_TCP_ACTION_NAMES
        ):
            self.raw_action_feature_names = list(DEFAULT_ROMOYA_TCP_ACTION_NAMES)
        elif raw_action_feature.shape[0] == transformed_action_dim and not self.raw_action_feature_names:
            self.raw_action_feature_names = (
                list(DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES) if is_tcp_mode else list(DEFAULT_ROMOYA_ACTION_NAMES)
            )

        raw_action_name_set = set(self.raw_action_feature_names)
        has_required_action_names = required_action_names.issubset(raw_action_name_set)
        is_transformed_action_shape = raw_action_feature.shape[0] == transformed_action_dim
        is_named_raw_action_shape = raw_action_feature.shape[0] == len(self.raw_action_feature_names)
        is_runtime_control_shape = (
            raw_action_feature.shape[0] >= len(DEFAULT_ROMOYA_ACTION_NAMES)
            and has_required_action_names
            and not is_named_raw_action_shape
        )

        if is_tcp_mode and not has_required_action_names and not is_transformed_action_shape:
            raise ValueError(
                "TCP action modes require datasets whose raw action contains tcp.* together with gripper.pos and DO_1. "
                "Convert the old 10D dataset before using a TCP action mode."
            )

        missing_joint_action = set(self.joint_action_names) - set(self.raw_action_feature_names)
        if missing_joint_action:
            raise ValueError(f"Unknown joint action names: {sorted(missing_joint_action)}")
        if self.gripper_action_name not in self.raw_action_feature_names:
            raise ValueError(f"Unknown gripper action name: {self.gripper_action_name}")

        missing_do = set(self.do_action_names) - set(self.raw_action_feature_names)
        if missing_do:
            raise ValueError(f"Unknown DO action names: {sorted(missing_do)}")
        if is_tcp_mode:
            missing_tcp = required_tcp_names - raw_action_name_set
            if missing_tcp:
                raise ValueError(
                    "TCP action mode requires raw action names including tcp.*, gripper.pos, and DO_1. "
                    f"Missing: {sorted(missing_tcp)}"
                )

        raw_action_dim = len(self.raw_action_feature_names)

        is_runtime_wide_state_shape = (
            raw_state_feature.shape[0] > raw_state_dim
            and required_state_names.issubset(set(self.raw_observation_state_feature_names))
        )

        if raw_state_feature.shape[0] not in (raw_state_dim, transformed_state_dim) and not is_runtime_wide_state_shape:
            raise ValueError(
                "Dataset observation.state shape does not match act_romoya raw or transformed state dimensions."
            )
        if not (is_named_raw_action_shape or is_transformed_action_shape or is_runtime_control_shape):
            raise ValueError(
                "Dataset action shape does not match act_romoya raw or transformed action dimensions."
            )

        # When creating a policy from dataset metadata we receive raw dataset shapes and need
        # to shrink them to the transformed Romoya state/action dimensions. When loading a
        # saved checkpoint config, the transformed dimensions are already stored and should be
        # preserved as-is.
        if raw_state_feature.shape[0] == raw_state_dim or is_runtime_wide_state_shape:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(transformed_state_dim,),
            )
        if is_named_raw_action_shape or is_runtime_control_shape:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(transformed_action_dim,),
            )

        super().validate_features()

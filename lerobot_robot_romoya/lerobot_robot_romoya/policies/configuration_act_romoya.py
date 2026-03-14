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


@ACTConfig.register_subclass("act_romoya")
@dataclass
class ACTRomoyaConfig(ACTConfig):
    chunk_size: int = 60
    n_action_steps: int = 60
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
    action_mode: str = "delta_joint_gripper_do1"
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
    sigmoid_do_outputs: bool = True

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    @property
    def transformed_action_names(self) -> list[str]:
        return [
            *(f"delta_{name}" for name in self.joint_action_names),
            f"delta_{self.gripper_action_name}",
            *self.do_action_names,
        ]

    def validate_features(self) -> None:
        if OBS_STATE not in (self.input_features or {}):
            raise ValueError("act_romoya requires observation.state in input_features.")
        if ACTION not in (self.output_features or {}):
            raise ValueError("act_romoya requires action in output_features.")
        if self.action_mode != "delta_joint_gripper_do1":
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        missing_state = set(self.state_feature_names_to_keep) - set(self.raw_observation_state_feature_names)
        if missing_state:
            raise ValueError(f"Unknown state feature names: {sorted(missing_state)}")

        missing_joint_action = set(self.joint_action_names) - set(self.raw_action_feature_names)
        if missing_joint_action:
            raise ValueError(f"Unknown joint action names: {sorted(missing_joint_action)}")
        if self.gripper_action_name not in self.raw_action_feature_names:
            raise ValueError(f"Unknown gripper action name: {self.gripper_action_name}")

        missing_do = set(self.do_action_names) - set(self.raw_action_feature_names)
        if missing_do:
            raise ValueError(f"Unknown DO action names: {sorted(missing_do)}")

        raw_state_feature = self.input_features[OBS_STATE]
        raw_action_feature = self.output_features[ACTION]
        if raw_state_feature.shape[0] != len(self.raw_observation_state_feature_names):
            raise ValueError(
                "Dataset observation.state shape does not match act_romoya raw_observation_state_feature_names."
            )
        if raw_action_feature.shape[0] != len(self.raw_action_feature_names):
            raise ValueError("Dataset action shape does not match act_romoya raw_action_feature_names.")

        self.input_features[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(len(self.state_feature_names_to_keep),),
        )
        self.output_features[ACTION] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(len(self.transformed_action_names),),
        )

        super().validate_features()

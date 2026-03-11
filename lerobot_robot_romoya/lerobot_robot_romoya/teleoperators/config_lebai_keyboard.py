from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("romoya_lebai_keyboard")
@dataclass
class LebaiKeyboardTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True
    use_suction: bool = True
    translation_step_m: float = 0.01
    rotation_step_rad: float = 0.1
    gripper_step: float = 10.0

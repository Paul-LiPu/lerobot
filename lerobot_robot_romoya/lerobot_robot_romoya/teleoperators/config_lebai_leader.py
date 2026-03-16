from dataclasses import dataclass

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("romoya_lebai_leader")
@dataclass
class LebaiLeaderConfig(TeleoperatorConfig):
    ip: str = "0.0.0.0"
    port: int | None = None
    simu: bool = False
    enter_teach_mode_on_connect: bool = True
    exit_teach_mode_on_disconnect: bool = True
    use_gripper: bool = True
    gripper_force: float = 100.0
    gripper_open_position: float = 99.0
    use_do: bool = True

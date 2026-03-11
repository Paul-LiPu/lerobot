from dataclasses import dataclass

from lerobot.robots import RobotConfig

from .config_lebai_follower import LebaiFollowerConfig


@RobotConfig.register_subclass("romoya_lebai_delta_tcp_follower")
@dataclass
class LebaiDeltaTcpFollowerConfig(LebaiFollowerConfig):
    cartesian_step_m: float = 0.01
    rotation_step_rad: float = 0.1
    gripper_step: float = 10.0


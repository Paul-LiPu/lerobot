from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("romoya_lebai_follower")
@RobotConfig.register_subclass("romoya_lebai")
@dataclass
class LebaiFollowerConfig(RobotConfig):
    ip: str = "10.20.17.1"
    port: int | None = None
    simu: bool = False
    acceleration: float = 1.0
    velocity: float = 1.0
    blend_radius: float = 0.0
    gripper_force: float = 100.0
    gripper_open_position: float = 99.0
    gripper_closed_position: float = 0.0
    default_do0: int = 0
    default_do1: int = 0
    use_effort: bool = True
    use_velocity: bool = True
    use_acceleration: bool = True
    trace_path: Path | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


LebaiRobotConfig = LebaiFollowerConfig

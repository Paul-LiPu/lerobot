import copy
from typing import Any

from lerobot.processor import RobotAction
from lerobot.utils.errors import DeviceNotConnectedError

from .config_lebai_tcp_follower import LebaiTcpFollowerConfig
from .lebai_follower import LebaiFollower


class LebaiTcpFollower(LebaiFollower):
    config_class = LebaiTcpFollowerConfig
    name = "romoya_lebai_tcp_follower"

    def __init__(self, config: LebaiTcpFollowerConfig):
        super().__init__(config)
        self.config = config

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "gripper": float,
            "DO_0": float,
            "DO_1": float,
        }

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        sent_action = copy.deepcopy(action)
        delta_x = float(sent_action.get("delta_x", 0.0))
        delta_y = float(sent_action.get("delta_y", 0.0))
        delta_z = float(sent_action.get("delta_z", 0.0))

        if delta_x or delta_y or delta_z:
            current_pose = self.get_tcp_pose()
            target_pose = dict(current_pose)
            target_pose["x"] = float(current_pose["x"] + delta_x)
            target_pose["y"] = float(current_pose["y"] + delta_y)
            target_pose["z"] = float(current_pose["z"] + delta_z)

            joint_targets = self.kinematics_inverse(target_pose)
            self.arm.movej(
                joint_targets,
                self.config.acceleration,
                self.config.velocity,
                0.0,
                self.config.blend_radius,
            )

        if "gripper" in sent_action:
            claw_data = self.get_claw_data()
            current_amplitude = float(claw_data.get("amplitude", 0.0))
            gripper_cmd = float(sent_action["gripper"])
            target_amplitude = max(0.0, min(100.0, current_amplitude + gripper_cmd))
            self.set_claw(target_amplitude, force=self.config.gripper_force)

        do0 = sent_action.get("DO_0")
        do1 = sent_action.get("DO_1")
        if do0 is not None:
            self.set_do("DO_0", 0, int(bool(do0)))
        if do1 is not None:
            self.set_do("DO_1", 1, int(bool(do1)))

        return sent_action


LebaiEEFollower = LebaiTcpFollower

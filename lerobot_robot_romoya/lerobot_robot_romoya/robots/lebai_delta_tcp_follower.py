import copy

from lerobot.processor import RobotAction
from lerobot.utils.errors import DeviceNotConnectedError

from .config_lebai_delta_tcp_follower import LebaiDeltaTcpFollowerConfig
from .lebai_follower import LebaiFollower


class LebaiDeltaTcpFollower(LebaiFollower):
    config_class = LebaiDeltaTcpFollowerConfig
    name = "romoya_lebai_delta_tcp_follower"

    def __init__(self, config: LebaiDeltaTcpFollowerConfig):
        super().__init__(config)
        self.config = config

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_wx": float,
            "delta_wy": float,
            "delta_wz": float,
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
        delta_wx = float(sent_action.get("delta_wx", 0.0))
        delta_wy = float(sent_action.get("delta_wy", 0.0))
        delta_wz = float(sent_action.get("delta_wz", 0.0))

        if delta_x or delta_y or delta_z or delta_wx or delta_wy or delta_wz:
            current_pose = self.get_tcp_pose()
            target_pose = dict(current_pose)
            target_pose["x"] = float(current_pose["x"] + delta_x)
            target_pose["y"] = float(current_pose["y"] + delta_y)
            target_pose["z"] = float(current_pose["z"] + delta_z)
            target_pose["rx"] = float(current_pose["rx"] + delta_wx)
            target_pose["ry"] = float(current_pose["ry"] + delta_wy)
            target_pose["rz"] = float(current_pose["rz"] + delta_wz)

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

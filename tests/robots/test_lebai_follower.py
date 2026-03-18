from lerobot_robot_romoya.policies.configuration_act_romoya import (
    DEFAULT_ROMOYA_OBS_STATE_NAMES,
    DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES,
)
from lerobot_robot_romoya.robots.config_lebai_follower import LebaiFollowerConfig
from lerobot_robot_romoya.robots.lebai_follower import ADDITIONAL_STATE_FEATURES, LebaiFollower


class _FakeArm:
    def get_kin_data(self):
        return {
            "actual_joint_pose": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "actual_joint_torque": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "actual_joint_speed": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            "actual_joint_acc": [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            "actual_tcp_pose": {"x": 0.1, "y": 0.2, "z": 0.3, "rx": 0.4, "ry": 0.5, "rz": 0.6},
            "actual_flange_pose": {"x": 0.9, "y": 0.8, "z": 0.7, "rx": 0.6, "ry": 0.5, "rz": 0.4},
            "target_joint_pose": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "target_joint_speed": [1.2, 1.1, 1.0, 0.9, 0.8, 0.7],
            "target_joint_acc": [1.8, 1.7, 1.6, 1.5, 1.4, 1.3],
            "target_joint_torque": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "target_tcp_pose": {"x": 1.1, "y": 1.2, "z": 1.3, "rx": 1.4, "ry": 1.5, "rz": 1.6},
        }

    def get_phy_data(self):
        return {
            "joint_temp": [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
            "joint_voltage": [48.1, 48.2, 48.3, 48.4, 48.5, 48.6],
            "flange_voltage": 23.4,
        }

    def get_claw_data(self):
        return {"force": 80.0, "amplitude": 90.0, "hold_on": False}

    def get_do(self, device, pin):
        if device == "DO_0":
            return 0
        if device == "DO_1":
            return 1
        raise AssertionError((device, pin))


def test_lebai_follower_observation_features_append_wide_state():
    robot = LebaiFollower(LebaiFollowerConfig(cameras={}))

    state_keys = [key for key, value in robot.observation_features.items() if value is float]

    assert state_keys[: len(DEFAULT_ROMOYA_OBS_STATE_NAMES)] == DEFAULT_ROMOYA_OBS_STATE_NAMES
    assert state_keys == DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES
    assert state_keys[-len(ADDITIONAL_STATE_FEATURES) :] == ADDITIONAL_STATE_FEATURES


def test_lebai_follower_action_features_use_10d_robot_control_schema():
    robot = LebaiFollower(LebaiFollowerConfig(cameras={}))

    action_keys = list(robot.action_features)

    assert action_keys == [
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


def test_lebai_follower_get_observation_includes_wide_state():
    robot = LebaiFollower(LebaiFollowerConfig(cameras={}))
    robot._arm = _FakeArm()

    obs = robot.get_observation()

    assert obs["joint1.temp"] == 31.0
    assert obs["joint6.voltage"] == 48.6
    assert obs["flange_voltage"] == 23.4
    assert obs["flange.x"] == 0.9
    assert obs["target_joint1.pos"] == 6.0
    assert obs["target_joint6.vel"] == 0.7
    assert obs["target_joint3.acc"] == 1.6
    assert obs["target_joint2.effort"] == 0.5
    assert obs["target_tcp.rz"] == 1.6

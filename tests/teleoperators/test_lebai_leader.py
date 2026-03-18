from lerobot_robot_romoya.teleoperators.config_lebai_leader import LebaiLeaderConfig
from lerobot_robot_romoya.teleoperators.lebai_leader import (
    BOTTOM_FACE_BUTTONS,
    RIGHT_FACE_BUTTONS,
    TOP_FACE_BUTTONS,
    LebaiLeader,
)


def test_lebai_leader_dpad_events_are_one_shot():
    teleop = LebaiLeader(LebaiLeaderConfig())

    teleop._handle_hat_value((1, 0))
    assert teleop.consume_control_events() == {"exit_early"}
    assert teleop.consume_control_events() == set()

    teleop._handle_hat_value((1, 0))
    assert teleop.consume_control_events() == set()

    teleop._handle_hat_value((0, 0))
    teleop._handle_hat_value((-1, 0))
    assert teleop.consume_control_events() == {"exit_early", "rerecord_episode"}

    teleop._handle_hat_value((0, 0))
    teleop._handle_hat_value((0, 1))
    assert teleop.consume_control_events() == {"save_reset_state"}

    teleop._handle_hat_value((0, 0))
    teleop._handle_hat_value((0, -1))
    assert teleop.consume_control_events() == {"clear_reset_state"}


def test_lebai_leader_dpad_debounces_noisy_center_repeat():
    teleop = LebaiLeader(LebaiLeaderConfig())

    teleop._handle_hat_value((0, 1))
    assert teleop.consume_control_events() == {"save_reset_state"}

    teleop._handle_hat_value((0, 0))
    teleop._handle_hat_value((0, 1))
    assert teleop.consume_control_events() == set()


def test_lebai_leader_button_aliases_toggle_gripper_and_suction():
    teleop = LebaiLeader(LebaiLeaderConfig())
    assert BOTTOM_FACE_BUTTONS == {0}
    assert RIGHT_FACE_BUTTONS == {1}
    assert TOP_FACE_BUTTONS == {3}

    assert teleop.gripper_target == 99.0
    teleop._toggle_gripper()
    assert teleop.gripper_target == 0.0
    teleop._toggle_gripper()
    assert teleop.gripper_target == 99.0

    assert teleop.do0_target == 0.0
    assert teleop.do1_target == 0.0
    teleop._toggle_suction()
    assert teleop.do0_target == 1.0
    assert teleop.do1_target == 1.0


def test_lebai_leader_top_face_button_loads_reset_state():
    teleop = LebaiLeader(LebaiLeaderConfig())
    teleop._pending_control_events.clear()
    teleop._pending_control_events.add("load_reset_state")
    assert teleop.consume_control_events() == {"load_reset_state"}


class _FakeLeaderArm:
    def get_kin_data(self):
        return {
            "actual_joint_pose": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "actual_joint_torque": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "actual_joint_speed": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            "actual_joint_acc": [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            "actual_tcp_pose": {"x": 0.1, "y": 0.2, "z": 0.3, "rx": 0.4, "ry": 0.5, "rz": 0.6},
        }

    def get_phy_data(self):
        return {
            "joint_temp": [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
            "joint_voltage": [48.1, 48.2, 48.3, 48.4, 48.5, 48.6],
            "flange_voltage": 23.4,
        }


def test_lebai_leader_action_includes_temp_voltage_and_flange_voltage():
    teleop = LebaiLeader(LebaiLeaderConfig())
    teleop._arm = _FakeLeaderArm()

    assert "joint1.temp" in teleop.action_features
    assert "joint6.voltage" in teleop.action_features
    assert "flange_voltage" in teleop.action_features

    action = teleop.get_action()

    assert action["joint1.temp"] == 31.0
    assert action["joint6.voltage"] == 48.6
    assert action["flange_voltage"] == 23.4

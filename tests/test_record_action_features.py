from lerobot.scripts.lerobot_record import _get_record_action_features
from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig


def test_record_action_features_use_teleop_for_teleop_only_recording():
    robot = MockRobot(MockRobotConfig(n_motors=2))
    teleop = MockTeleop(MockTeleopConfig(n_motors=3))

    features = _get_record_action_features(robot, teleop, has_policy=False)

    assert list(features) == ["motor_1.pos", "motor_2.pos", "motor_3.pos"]


def test_record_action_features_use_robot_when_policy_is_present():
    robot = MockRobot(MockRobotConfig(n_motors=2))
    teleop = MockTeleop(MockTeleopConfig(n_motors=3))

    features = _get_record_action_features(robot, teleop, has_policy=True)

    assert list(features) == ["motor_1.pos", "motor_2.pos"]

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import (
    DatasetRecordConfig,
    RecordConfig,
    _apply_saved_reset_state,
    _capture_reset_state_from_observation,
    _load_initial_pose,
    _save_initial_pose,
    record,
)
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_teleoperate():
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


def test_record_and_replay(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)


def test_capture_reset_state_from_observation():
    observation = {
        "joint1.pos": 1.0,
        "joint2.pos": 2.0,
        "joint3.pos": 3.0,
        "joint4.pos": 4.0,
        "joint5.pos": 5.0,
        "joint6.pos": 6.0,
        "gripper.pos": 90.0,
        "DO_0": 0.0,
        "DO_1": 1.0,
        "tcp.x": 0.1,
    }

    state = _capture_reset_state_from_observation(observation)

    assert state == {
        "joint1.pos": 1.0,
        "joint2.pos": 2.0,
        "joint3.pos": 3.0,
        "joint4.pos": 4.0,
        "joint5.pos": 5.0,
        "joint6.pos": 6.0,
        "gripper.pos": 90.0,
        "DO_0": 0.0,
        "DO_1": 1.0,
    }


def test_apply_saved_reset_state():
    class DummyResetRobot:
        name = "romoya_lebai_follower"

        def __init__(self):
            self.joints = None
            self.action = None

        def move_to_joint_positions(self, joint_positions):
            self.joints = list(joint_positions)

        def send_action(self, action):
            self.action = dict(action)

    class DummyLeaderTeleop:
        def __init__(self):
            self.joints = None
            self.saved_state = None

        def move_to_joint_positions(self, joint_positions):
            self.joints = list(joint_positions)

        def apply_saved_control_state(self, saved_state):
            self.saved_state = dict(saved_state)

    robot = DummyResetRobot()
    teleop = DummyLeaderTeleop()
    saved_state = {
        "joint1.pos": 1.0,
        "joint2.pos": 2.0,
        "joint3.pos": 3.0,
        "joint4.pos": 4.0,
        "joint5.pos": 5.0,
        "joint6.pos": 6.0,
        "gripper.pos": 90.0,
        "DO_0": 0.0,
        "DO_1": 1.0,
    }

    applied = _apply_saved_reset_state(robot, teleop, saved_state)

    assert applied is True
    assert teleop.joints == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert teleop.saved_state == saved_state
    assert robot.joints is None
    assert robot.action is None


def test_save_and_load_initial_pose(tmp_path):
    saved_state = {
        "joint1.pos": 1.0,
        "joint2.pos": 2.0,
        "joint3.pos": 3.0,
        "joint4.pos": 4.0,
        "joint5.pos": 5.0,
        "joint6.pos": 6.0,
        "gripper.pos": 90.0,
        "DO_0": 0.0,
        "DO_1": 1.0,
    }
    pose_path = tmp_path / "initial_pose.json"

    assert _save_initial_pose(pose_path, saved_state) is True
    assert _load_initial_pose(pose_path) == saved_state


def test_load_initial_pose_returns_none_for_missing_file(tmp_path):
    assert _load_initial_pose(tmp_path / "missing_initial_pose.json") is None

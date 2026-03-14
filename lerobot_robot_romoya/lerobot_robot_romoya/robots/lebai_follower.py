import copy
import logging
from typing import Any

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..lebai_sdk_utils import JOINT_COUNT, TCP_KEYS, connect_arm, get_claw_data, get_field, maybe_read_camera, pose_to_dict
from .config_lebai_follower import LebaiFollowerConfig

logger = logging.getLogger(__name__)


class LebaiFollower(Robot):
    config_class = LebaiFollowerConfig
    name = "romoya_lebai_follower"

    def __init__(self, config: LebaiFollowerConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._last_gripper_target: float | None = None
        self._last_gripper_force: float | None = None
        self._last_do0_target: int | None = None
        self._last_do1_target: int | None = None

        from lerobot.cameras import make_cameras_from_configs

        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def arm(self) -> Any:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._arm

    @property
    def _motor_features(self) -> dict[str, type]:
        features = {f"joint{i}.pos": float for i in range(1, JOINT_COUNT + 1)}
        features["gripper.pos"] = float
        features["gripper.force"] = float
        features["DO_0"] = float
        features["DO_1"] = float
        return features

    @property
    def _action_features(self) -> dict[str, type]:
        features = dict(self._motor_features)
        features["gripper.force"] = float
        features["DO_0"] = float
        features["DO_1"] = float
        return features

    @property
    def _camera_features(self) -> dict[str, tuple[int, int, int]]:
        return {
            cam_key: (self.config.cameras[cam_key].height, self.config.cameras[cam_key].width, 3)
            for cam_key in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        features = {**self._motor_features, **self._camera_features}
        if self.config.use_effort:
            for i in range(1, JOINT_COUNT + 1):
                features[f"joint{i}.effort"] = float
        if self.config.use_velocity:
            for i in range(1, JOINT_COUNT + 1):
                features[f"joint{i}.vel"] = float
        if self.config.use_acceleration:
            for i in range(1, JOINT_COUNT + 1):
                features[f"joint{i}.acc"] = float
        for key in TCP_KEYS:
            features[f"tcp.{key}"] = float
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return self._action_features

    @property
    def is_connected(self) -> bool:
        if self._arm is None:
            return False
        is_connected = getattr(self._arm, "is_connected", None)
        if callable(is_connected):
            return bool(is_connected())
        is_disconnected = getattr(self._arm, "is_disconnected", None)
        if callable(is_disconnected):
            return not bool(is_disconnected())
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")

        self._arm = connect_arm(self.config.ip, port=self.config.port, simu=self.config.simu)
        self.start_system()
        for camera in self.cameras.values():
            camera.connect()
        self.configure()
        self._last_gripper_target = float(get_claw_data(self.arm)["amplitude"])
        self._last_gripper_force = None
        self._last_do0_target = None
        self._last_do1_target = None
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        _ = self.get_tcp_pose()

    def _get_kin_data(self) -> Any:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self.arm.get_kin_data()

    def get_tcp_pose(self) -> dict[str, float]:
        return pose_to_dict(get_field(self._get_kin_data(), "actual_tcp_pose"))

    def get_observation(self) -> RobotObservation:
        kin_data = self._get_kin_data()
        obs: dict[str, Any] = {}

        joint_positions = list(get_field(kin_data, "actual_joint_pose"))
        joint_torque = list(get_field(kin_data, "actual_joint_torque"))
        joint_speed = list(get_field(kin_data, "actual_joint_speed"))
        joint_acc = list(get_field(kin_data, "actual_joint_acc"))

        for index, position in enumerate(joint_positions, start=1):
            obs[f"joint{index}.pos"] = float(position)
            if self.config.use_effort and index <= len(joint_torque):
                obs[f"joint{index}.effort"] = float(joint_torque[index - 1])
            if self.config.use_velocity and index <= len(joint_speed):
                obs[f"joint{index}.vel"] = float(joint_speed[index - 1])
            if self.config.use_acceleration and index <= len(joint_acc):
                obs[f"joint{index}.acc"] = float(joint_acc[index - 1])

        claw_data = get_claw_data(self.arm)
        obs["gripper.pos"] = float(claw_data["amplitude"])
        obs["gripper.force"] = float(claw_data["force"])
        obs["DO_0"] = float(self.get_do("DO_0", 0))
        obs["DO_1"] = float(self.get_do("DO_1", 1))

        for key, value in pose_to_dict(get_field(kin_data, "actual_tcp_pose")).items():
            obs[f"tcp.{key}"] = value

        for cam_key, camera in self.cameras.items():
            obs[cam_key] = maybe_read_camera(camera)

        return obs

    def _extract_joint_targets(self, action: RobotAction) -> list[float] | None:
        if not any(key.endswith(".pos") and key.startswith("joint") for key in action):
            return None

        current_joints = list(get_field(self._get_kin_data(), "actual_joint_pose"))
        joint_targets = current_joints[:JOINT_COUNT]
        for index in range(1, JOINT_COUNT + 1):
            key = f"joint{index}.pos"
            if key in action:
                joint_targets[index - 1] = float(action[key])
        return joint_targets

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        sent_action = copy.deepcopy(action)
        joint_targets = self._extract_joint_targets(sent_action)
        if joint_targets is not None:
            self.arm.towardj(
                joint_targets,
                self.config.acceleration,
                self.config.velocity,
                0.0,
                self.config.blend_radius,
            )

        if "gripper.pos" in sent_action:
            gripper_target = float(sent_action["gripper.pos"])
            gripper_force = float(sent_action.get("gripper.force", self.config.gripper_force))
            if abs(gripper_target) <= 1e-6:
                gripper_target = self.config.gripper_closed_position
            pos_changed = self._last_gripper_target is None or abs(gripper_target - self._last_gripper_target) > 1e-3
            force_changed = self._last_gripper_force is None or abs(gripper_force - self._last_gripper_force) > 1e-3
            if pos_changed or force_changed:
                self.set_claw(gripper_target, force=gripper_force)
                self._last_gripper_target = gripper_target
                self._last_gripper_force = gripper_force
        do0 = sent_action.get("DO_0")
        do1 = sent_action.get("DO_1")
        if do0 is not None:
            do0_target = int(bool(do0))
            if self._last_do0_target is None or do0_target != self._last_do0_target:
                self.set_do("DO_0", 0, do0_target)
                self._last_do0_target = do0_target
        if do1 is not None:
            do1_target = int(bool(do1))
            if self._last_do1_target is None or do1_target != self._last_do1_target:
                self.set_do("DO_1", 1, do1_target)
                self._last_do1_target = do1_target

        return sent_action

    def _set_safe_end_effector_state(self) -> None:
        """Best-effort safe EE shutdown without interrupting disconnect."""
        try:
            self.set_claw(self.config.gripper_open_position, force=self.config.gripper_force)
            self._last_gripper_target = self.config.gripper_open_position
            self._last_gripper_force = self.config.gripper_force
        except Exception:
            logger.warning("%s failed to reset gripper during disconnect.", self, exc_info=True)

        try:
            self.set_do("DO_0", 0, self.config.default_do0)
            self._last_do0_target = int(self.config.default_do0)
        except Exception:
            logger.warning("%s failed to reset DO_0 during disconnect.", self, exc_info=True)

        try:
            self.set_do("DO_1", 1, self.config.default_do1)
            self._last_do1_target = int(self.config.default_do1)
        except Exception:
            logger.warning("%s failed to reset DO_1 during disconnect.", self, exc_info=True)

    def disconnect(self) -> None:
        if self._arm is not None:
            try:
                self._set_safe_end_effector_state()
            finally:
                self.stop_system()
                self._arm = None
                self._last_gripper_target = None
                self._last_gripper_force = None
                self._last_do0_target = None
                self._last_do1_target = None

        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except Exception:
                logger.exception("Failed disconnecting camera")

        logger.info("%s disconnected.", self)

    def wait_move(self) -> None:
        wait_move = getattr(self.arm, "wait_move", None)
        if callable(wait_move):
            wait_move()

    def start_system(self) -> None:
        start_sys = getattr(self.arm, "start_sys", None)
        if callable(start_sys):
            start_sys()

    def stop_system(self) -> None:
        if not self.is_connected:
            return
        stop_sys = getattr(self.arm, "stop_sys", None)
        if callable(stop_sys):
            stop_sys()

    def pause_move(self) -> None:
        pause_move = getattr(self.arm, "pause_move", None)
        if callable(pause_move):
            pause_move()

    def resume_move(self) -> None:
        resume_move = getattr(self.arm, "resume_move", None)
        if callable(resume_move):
            resume_move()

    def set_do(self, device: str, pin: int, value: int | bool) -> None:
        self.arm.set_do(device, int(pin), int(value))

    def get_do(self, device: str, pin: int) -> int:
        getter = getattr(self.arm, "get_do", None)
        if callable(getter):
            try:
                return int(getter(device, int(pin)))
            except TypeError:
                pass
            except Exception:
                pass

        if device == "DO_0":
            return int(self._last_do0_target or 0)
        if device == "DO_1":
            return int(self._last_do1_target or 0)
        return 0

    def set_claw(self, amplitude: float, force: float | None = None) -> None:
        self.arm.set_claw(float(force if force is not None else self.config.gripper_force), float(amplitude))

    def get_claw_data(self) -> dict[str, float | bool]:
        return get_claw_data(self.arm)

    def kinematics_inverse(self, target_pose: dict[str, float], seed_joints: list[float] | None = None) -> list[float]:
        seed = seed_joints if seed_joints is not None else list(get_field(self._get_kin_data(), "actual_joint_pose"))
        result = self.arm.kinematics_inverse(dict(target_pose), seed)
        if hasattr(result, "joint_positions"):
            return list(result.joint_positions)
        return list(result)

    def move_to_joint_positions(
        self,
        joint_positions: list[float],
        *,
        acceleration: float | None = None,
        velocity: float | None = None,
        blend_radius: float | None = None,
        wait: bool = True,
    ) -> None:
        self.arm.towardj(
            joint_positions,
            self.config.acceleration if acceleration is None else acceleration,
            self.config.velocity if velocity is None else velocity,
            0.0,
            self.config.blend_radius if blend_radius is None else blend_radius,
        )
        if wait:
            self.wait_move()

    def move_to_tcp_pose(
        self,
        tcp_pose: dict[str, float],
        *,
        acceleration: float | None = None,
        velocity: float | None = None,
        blend_radius: float | None = None,
        wait: bool = True,
    ) -> None:
        self.arm.movel(
            dict(tcp_pose),
            self.config.acceleration if acceleration is None else acceleration,
            self.config.velocity if velocity is None else velocity,
            0.0,
            self.config.blend_radius if blend_radius is None else blend_radius,
        )
        if wait:
            self.wait_move()


LebaiRobot = LebaiFollower

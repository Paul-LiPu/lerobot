import logging
import os
import sys
from queue import Queue
import time
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..lebai_sdk_utils import JOINT_COUNT, connect_arm, get_field
from .config_lebai_leader import LebaiLeaderConfig

logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")
    from pynput import keyboard, mouse
except ImportError:
    keyboard = None
    mouse = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    mouse = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


class LebaiLeader(Teleoperator):
    config_class = LebaiLeaderConfig
    name = "romoya_lebai_leader"

    def __init__(self, config: LebaiLeaderConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self.event_queue = Queue()
        self.current_pressed = {}
        self.keyboard_listener = None
        self.mouse_listener = None
        self.gripper_target = 100.0
        self.do0_target = 0.0
        self.do1_target = 0.0
        self._gripper_open = True
        self._suction_on = False
        self.logs = {}

    @property
    def arm(self) -> Any:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._arm

    @property
    def action_features(self) -> dict[str, type]:
        features = {f"joint{i}.pos": float for i in range(1, JOINT_COUNT + 1)}
        if self.config.use_gripper:
            features["gripper.pos"] = float
            features["gripper.force"] = float
        if self.config.use_do:
            features["DO_0"] = float
            features["DO_1"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

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
        start_sys = getattr(self.arm, "start_sys", None)
        if callable(start_sys):
            start_sys()
        if self.config.enter_teach_mode_on_connect:
            self.teach_mode()
        if PYNPUT_AVAILABLE:
            self.keyboard_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self.keyboard_listener.start()
            self.mouse_listener = mouse.Listener(on_click=self._on_click, on_scroll=self._on_scroll)
            self.mouse_listener.start()
        else:
            self.keyboard_listener = None
            self.mouse_listener = None
        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _on_press(self, key) -> None:
        self.event_queue.put((key, True))

    def _on_release(self, key) -> None:
        self.event_queue.put((key, False))
        if PYNPUT_AVAILABLE and key == keyboard.Key.esc:
            logger.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _on_click(self, x, y, button, pressed) -> None:
        del x, y
        if not pressed or not PYNPUT_AVAILABLE:
            return
        if button == mouse.Button.left:
            self._gripper_open = not self._gripper_open
            self.gripper_target = 100.0 if self._gripper_open else 0.0
        elif button == mouse.Button.right:
            self._suction_on = not self._suction_on
            if self._suction_on:
                self.do0_target = 1.0
                self.do1_target = 1.0
            else:
                self.do0_target = 0.0
                self.do1_target = 0.0

    def _on_scroll(self, x, y, dx, dy) -> None:
        del x, y, dx, dy
        return None

    def _drain_pressed_keys(self) -> None:
        while not self.event_queue.empty():
            key, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key] = is_pressed
            if not is_pressed:
                continue

            if key == keyboard.Key.right:
                self.gripper_target = 100.0
                self._gripper_open = True
            elif key == keyboard.Key.left:
                self.gripper_target = 0.0
                self._gripper_open = False
            elif key == keyboard.KeyCode.from_char("1"):
                self.do0_target = 1.0
                self.do1_target = 1.0
                self._suction_on = True
            elif key == keyboard.KeyCode.from_char("2"):
                self.do0_target = 0.0
                self.do1_target = 1.0
                self._suction_on = False
            elif key == keyboard.KeyCode.from_char("3"):
                self.do0_target = 0.0
                self.do1_target = 0.0
                self._suction_on = False

    def get_action(self) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        before_read_t = time.perf_counter()
        self._drain_pressed_keys()
        kin_data = self.arm.get_kin_data()
        action = {}
        joint_positions = list(get_field(kin_data, "actual_joint_pose"))
        for index, position in enumerate(joint_positions, start=1):
            action[f"joint{index}.pos"] = float(position)
        if self.config.use_gripper:
            action["gripper.pos"] = self.gripper_target
            action["gripper.force"] = float(self.config.gripper_force)
        if self.config.use_do:
            action["DO_0"] = self.do0_target
            action["DO_1"] = self.do1_target
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    def teach_mode(self) -> None:
        teach_mode = getattr(self.arm, "teach_mode", None)
        if callable(teach_mode):
            teach_mode()

    def end_teach_mode(self) -> None:
        end_teach_mode = getattr(self.arm, "end_teach_mode", None)
        if callable(end_teach_mode):
            end_teach_mode()

    def disconnect(self) -> None:
        if self._arm is None:
            return
        try:
            if self.keyboard_listener is not None:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
            if self.mouse_listener is not None:
                self.mouse_listener.stop()
                self.mouse_listener = None
            if self.config.exit_teach_mode_on_disconnect:
                self.end_teach_mode()
            stop_sys = getattr(self.arm, "stop_sys", None)
            if callable(stop_sys):
                stop_sys()
        finally:
            self._arm = None
        logger.info("%s disconnected.", self)

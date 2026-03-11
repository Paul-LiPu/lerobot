import logging
import os
import sys
from queue import Queue
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_lebai_keyboard import LebaiKeyboardTeleopConfig

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as exc:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info("Could not import pynput: %s", exc)


logger = logging.getLogger(__name__)


class LebaiKeyboardTeleop(Teleoperator):
    config_class = LebaiKeyboardTeleopConfig
    name = "romoya_lebai_keyboard"

    def __init__(self, config: LebaiKeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.event_queue = Queue()
        self.current_pressed: dict[Any, bool] = {}
        self.listener = None
        self._do0 = 0.0
        self._do1 = 0.0

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_wx": float,
            "delta_wy": float,
            "delta_wz": float,
        }
        if self.config.use_gripper:
            features["gripper"] = float
        if self.config.use_suction:
            features["DO_0"] = float
            features["DO_1"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if not PYNPUT_AVAILABLE:
            raise ImportError("pynput is required for romoya_lebai_keyboard teleop.")
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _on_press(self, key: Any) -> None:
        self.event_queue.put((key, True))

    def _on_release(self, key: Any) -> None:
        self.event_queue.put((key, False))
        if key == keyboard.Key.esc:
            logger.info("ESC pressed, disconnecting.")
            self.disconnect()
        elif hasattr(key, "char"):
            if key.char == "1":
                self._do0 = 1.0
                self._do1 = 1.0
            elif key.char == "2":
                self._do0 = 0.0
                self._do1 = 1.0
            elif key.char == "3":
                self._do0 = 0.0
                self._do1 = 0.0

    def _drain_pressed_keys(self) -> None:
        while not self.event_queue.empty():
            key, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key] = is_pressed

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        action: RobotAction = {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "delta_wx": 0.0,
            "delta_wy": 0.0,
            "delta_wz": 0.0,
        }

        for key, val in self.current_pressed.items():
            if not val:
                continue
            if key == keyboard.Key.up:
                action["delta_y"] = -self.config.translation_step_m
            elif key == keyboard.Key.down:
                action["delta_y"] = self.config.translation_step_m
            elif key == keyboard.Key.left:
                action["delta_x"] = self.config.translation_step_m
            elif key == keyboard.Key.right:
                action["delta_x"] = -self.config.translation_step_m
            elif key == keyboard.Key.shift:
                action["delta_z"] = -self.config.translation_step_m
            elif key == keyboard.Key.shift_r:
                action["delta_z"] = self.config.translation_step_m
            elif hasattr(key, "char"):
                if key.char == "w":
                    action["delta_wx"] = self.config.rotation_step_rad
                elif key.char == "s":
                    action["delta_wx"] = -self.config.rotation_step_rad
                elif key.char == "a":
                    action["delta_wy"] = self.config.rotation_step_rad
                elif key.char == "d":
                    action["delta_wy"] = -self.config.rotation_step_rad
                elif key.char == "q":
                    action["delta_wz"] = self.config.rotation_step_rad
                elif key.char == "e":
                    action["delta_wz"] = -self.config.rotation_step_rad

        if self.config.use_gripper:
            gripper = 0.0
            if self.current_pressed.get(keyboard.Key.ctrl_r, False):
                gripper = self.config.gripper_step
            elif self.current_pressed.get(keyboard.Key.ctrl_l, False):
                gripper = -self.config.gripper_step
            action["gripper"] = gripper

        if self.config.use_suction:
            action["DO_0"] = self._do0
            action["DO_1"] = self._do1

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()

import logging
import os
import sys
from queue import Queue
import time
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..lebai_sdk_utils import JOINT_COUNT, TCP_KEYS, connect_arm, get_field, get_float_sequence, pose_to_dict
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

PYGAME_AVAILABLE = True
try:
    import pygame
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False
except Exception as e:
    pygame = None
    PYGAME_AVAILABLE = False
    logging.info(f"Could not import pygame: {e}")

# Face-button mapping for common Xbox/PlayStation layouts.
# Bottom button:
# - Xbox A
# - PlayStation Cross
# Right button:
# - Xbox B
# - PlayStation Circle
BOTTOM_FACE_BUTTONS = {0}
RIGHT_FACE_BUTTONS = {1}
TOP_FACE_BUTTONS = {3}
DPAD_EVENT_MAP = {
    (1, 0): ("exit_early",),
    (-1, 0): ("rerecord_episode", "exit_early"),
    (0, 1): ("save_reset_state",),
    (0, -1): ("clear_reset_state",),
}
DPAD_DEBOUNCE_S = 1.0


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
        self.gamepad = None
        self._using_gamepad = False
        self.gripper_target = float(self.config.gripper_open_position)
        self.do0_target = 0.0
        self.do1_target = 0.0
        self._pending_control_events: set[str] = set()
        self._last_hat_value = (0, 0)
        self._last_hat_trigger_ts = 0.0
        self._logged_unknown_buttons: set[int] = set()
        self._gripper_open = True
        self._suction_on = False
        self._suction_stop_deadline: float | None = None
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
        if self.config.use_effort:
            for i in range(1, JOINT_COUNT + 1):
                features[f"joint{i}.effort"] = float
        if self.config.use_velocity:
            for i in range(1, JOINT_COUNT + 1):
                features[f"joint{i}.vel"] = float
        if self.config.use_acceleration:
            for i in range(1, JOINT_COUNT + 1):
                features[f"joint{i}.acc"] = float
        for i in range(1, JOINT_COUNT + 1):
            features[f"joint{i}.temp"] = float
            features[f"joint{i}.voltage"] = float
        features["flange_voltage"] = float
        for key in TCP_KEYS:
            features[f"tcp.{key}"] = float
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
            self._enter_teach_mode_with_retry()
        self._connect_gamepad()
        if PYNPUT_AVAILABLE and not self._using_gamepad:
            logger.info("Lebai leader control mode: keyboard/mouse fallback")
            self.keyboard_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self.keyboard_listener.start()
            self.mouse_listener = mouse.Listener(on_click=self._on_click, on_scroll=self._on_scroll)
            self.mouse_listener.start()
        else:
            self.keyboard_listener = None
            self.mouse_listener = None
            if self._using_gamepad:
                logger.info("Lebai leader control mode: gamepad")
            else:
                logger.warning("Lebai leader control mode: no gamepad and keyboard/mouse unavailable")
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
            self.gripper_target = float(self.config.gripper_open_position) if self._gripper_open else 0.0
        elif button == mouse.Button.right:
            now = time.monotonic()
            if self._suction_stop_deadline is not None and now < self._suction_stop_deadline:
                self._suction_stop_deadline = None
                self._suction_on = True
                self.do0_target = 1.0
                self.do1_target = 1.0
            elif self._suction_on:
                self._suction_stop_deadline = now + 1.0
                self.do0_target = 0.0
                self.do1_target = 1.0
            else:
                self._suction_stop_deadline = None
                self._suction_on = True
                self.do0_target = 1.0
                self.do1_target = 1.0

    def _on_scroll(self, x, y, dx, dy) -> None:
        del x, y, dx, dy
        return None

    def _toggle_gripper(self) -> None:
        self._gripper_open = not self._gripper_open
        self.gripper_target = float(self.config.gripper_open_position) if self._gripper_open else 0.0

    def _toggle_suction(self) -> None:
        now = time.monotonic()
        if self._suction_stop_deadline is not None and now < self._suction_stop_deadline:
            self._suction_stop_deadline = None
            self._suction_on = True
            self.do0_target = 1.0
            self.do1_target = 1.0
        elif self._suction_on:
            self._suction_stop_deadline = now + 1.0
            self.do0_target = 0.0
            self.do1_target = 1.0
        else:
            self._suction_stop_deadline = None
            self._suction_on = True
            self.do0_target = 1.0
            self.do1_target = 1.0

    def _drain_pressed_keys(self) -> None:
        while not self.event_queue.empty():
            key, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key] = is_pressed

    def _handle_hat_value(self, hat_value: tuple[int, int]) -> None:
        if hat_value == self._last_hat_value:
            return
        now = time.monotonic()
        if hat_value == (0, 0):
            self._last_hat_value = hat_value
            return
        if now - self._last_hat_trigger_ts < DPAD_DEBOUNCE_S:
            self._last_hat_value = hat_value
            return
        self._last_hat_value = hat_value
        self._last_hat_trigger_ts = now
        self._pending_control_events.update(DPAD_EVENT_MAP.get(hat_value, ()))

    def _connect_gamepad(self) -> None:
        self._using_gamepad = False
        if not (self.config.use_gamepad and PYGAME_AVAILABLE):
            if not self.config.use_gamepad:
                logger.info("Gamepad disabled by config for Lebai leader teleop")
            else:
                logger.info("pygame unavailable; cannot use gamepad for Lebai leader teleop")
            return
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            logger.info("No gamepad detected for Lebai leader teleop; falling back if keyboard is available")
            return
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self._using_gamepad = True
        logger.info(
            "Initialized leader gamepad: %s (buttons=%s hats=%s axes=%s)",
            self.gamepad.get_name(),
            self.gamepad.get_numbuttons(),
            self.gamepad.get_numhats(),
            self.gamepad.get_numaxes(),
        )

    def _disconnect_gamepad(self) -> None:
        if not PYGAME_AVAILABLE:
            return
        if self.gamepad is not None:
            self.gamepad.quit()
            self.gamepad = None
        self._using_gamepad = False
        if pygame.joystick.get_init():
            pygame.joystick.quit()
        if pygame.get_init():
            pygame.quit()

    def _drain_gamepad_events(self) -> None:
        if self.gamepad is None or not PYGAME_AVAILABLE:
            return
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button in BOTTOM_FACE_BUTTONS:
                    logger.info("Leader gamepad button %s -> toggle gripper", event.button)
                    self._toggle_gripper()
                elif event.button in RIGHT_FACE_BUTTONS:
                    logger.info("Leader gamepad button %s -> toggle suction", event.button)
                    self._toggle_suction()
                elif event.button in TOP_FACE_BUTTONS:
                    logger.info("Leader gamepad button %s -> load reset state", event.button)
                    self._pending_control_events.add("load_reset_state")
                elif event.button not in self._logged_unknown_buttons:
                    logger.info("Leader gamepad button %s pressed with no Romoya mapping", event.button)
                    self._logged_unknown_buttons.add(event.button)
            elif event.type == pygame.JOYHATMOTION:
                logger.info("Leader gamepad hat -> %s", tuple(event.value))
                self._handle_hat_value(tuple(event.value))

    def consume_control_events(self) -> set[str]:
        self._drain_gamepad_events()
        pending = set(self._pending_control_events)
        self._pending_control_events.clear()
        return pending

    def _update_suction_sequence(self) -> None:
        if self._suction_stop_deadline is None:
            return
        if time.monotonic() >= self._suction_stop_deadline:
            self._suction_stop_deadline = None
            self._suction_on = False
            self.do0_target = 0.0
            self.do1_target = 0.0

    def get_action(self) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        before_read_t = time.perf_counter()
        self._drain_pressed_keys()
        self._drain_gamepad_events()
        self._update_suction_sequence()
        kin_data = self.arm.get_kin_data()
        phy_data = self.arm.get_phy_data()
        action = {}
        joint_positions = get_float_sequence(kin_data, "actual_joint_pose", expected_len=JOINT_COUNT)
        joint_torque = get_float_sequence(kin_data, "actual_joint_torque", expected_len=JOINT_COUNT)
        joint_speed = get_float_sequence(kin_data, "actual_joint_speed", expected_len=JOINT_COUNT)
        joint_acc = get_float_sequence(kin_data, "actual_joint_acc", expected_len=JOINT_COUNT)
        joint_temp = get_float_sequence(phy_data, "joint_temp", expected_len=JOINT_COUNT)
        joint_voltage = get_float_sequence(phy_data, "joint_voltage", expected_len=JOINT_COUNT)
        for index, position in enumerate(joint_positions, start=1):
            action[f"joint{index}.pos"] = float(position)
            if self.config.use_effort and index <= len(joint_torque):
                action[f"joint{index}.effort"] = float(joint_torque[index - 1])
            if self.config.use_velocity and index <= len(joint_speed):
                action[f"joint{index}.vel"] = float(joint_speed[index - 1])
            if self.config.use_acceleration and index <= len(joint_acc):
                action[f"joint{index}.acc"] = float(joint_acc[index - 1])
            action[f"joint{index}.temp"] = joint_temp[index - 1]
            action[f"joint{index}.voltage"] = joint_voltage[index - 1]
        if self.config.use_gripper:
            action["gripper.pos"] = self.gripper_target
            action["gripper.force"] = float(self.config.gripper_force)
        if self.config.use_do:
            action["DO_0"] = self.do0_target
            action["DO_1"] = self.do1_target
        action["flange_voltage"] = float(get_field(phy_data, "flange_voltage"))
        for key, value in pose_to_dict(get_field(kin_data, "actual_tcp_pose")).items():
            action[f"tcp.{key}"] = value
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    def teach_mode(self) -> None:
        teach_mode = getattr(self.arm, "teach_mode", None)
        if callable(teach_mode):
            teach_mode()

    def _enter_teach_mode_with_retry(self, attempts: int = 10, retry_delay_s: float = 0.5) -> None:
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                self.teach_mode()
                return
            except Exception as exc:
                last_error = exc
                if "Moving" not in str(exc):
                    raise
                time.sleep(retry_delay_s)

        logger.warning("Failed to enter teach mode on connect after retries: %s", last_error)

    def end_teach_mode(self) -> None:
        end_teach_mode = getattr(self.arm, "end_teach_mode", None)
        if callable(end_teach_mode):
            end_teach_mode()

    def wait_move(self) -> None:
        wait_move = getattr(self.arm, "wait_move", None)
        if callable(wait_move):
            wait_move()

    def move_to_joint_positions(
        self,
        joint_positions: list[float],
        *,
        acceleration: float | None = None,
        velocity: float | None = None,
        blend_radius: float | None = None,
        wait: bool = True,
    ) -> None:
        reenter_teach_mode = self.config.enter_teach_mode_on_connect
        if reenter_teach_mode:
            try:
                self.end_teach_mode()
            except Exception:
                logger.exception("Failed exiting teach mode before moving leader.")

        try:
            self.arm.towardj(
                list(joint_positions),
                self.config.acceleration if acceleration is None else acceleration,
                self.config.velocity if velocity is None else velocity,
                0.0,
                self.config.blend_radius if blend_radius is None else blend_radius,
            )
            if wait:
                self.wait_move()
        finally:
            if reenter_teach_mode:
                try:
                    self._enter_teach_mode_with_retry()
                except Exception:
                    logger.exception("Failed re-entering teach mode after moving leader.")

    def apply_saved_control_state(self, saved_state: dict[str, float]) -> None:
        self.gripper_target = float(saved_state["gripper.pos"])
        self._gripper_open = self.gripper_target >= float(self.config.gripper_open_position) / 2.0
        self.do0_target = float(saved_state["DO_0"])
        self.do1_target = float(saved_state["DO_1"])
        self._suction_stop_deadline = None
        self._suction_on = bool(self.do0_target or self.do1_target)

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
            self._disconnect_gamepad()
            if self.config.exit_teach_mode_on_disconnect:
                self.end_teach_mode()
            stop_sys = getattr(self.arm, "stop_sys", None)
            if callable(stop_sys):
                stop_sys()
        finally:
            self._arm = None
        logger.info("%s disconnected.", self)

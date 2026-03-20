from collections.abc import Mapping
import importlib
import logging
import os
from typing import Any

JOINT_COUNT = 6
TCP_KEYS = ("x", "y", "z", "rx", "ry", "rz")
_NEST_ASYNCIO_APPLIED = False
logger = logging.getLogger(__name__)


def apply_nest_asyncio() -> None:
    global _NEST_ASYNCIO_APPLIED
    if _NEST_ASYNCIO_APPLIED:
        return

    try:
        nest_asyncio = importlib.import_module("nest_asyncio")
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The Romoya Lebai plugin requires 'nest_asyncio' for Lebai SDK compatibility. "
            "Install the Romoya plugin dependencies again so 'nest_asyncio' is available."
        )

    nest_asyncio.apply()
    _NEST_ASYNCIO_APPLIED = True
    logger.info("Applied nest_asyncio for Lebai SDK compatibility.")


def get_lebai_sdk() -> Any:
    apply_nest_asyncio()
    try:
        return importlib.import_module("lebai_sdk")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Romoya Lebai plugin requires the legacy 'lebai_sdk' package. "
            "Install 'lebai-sdk==0.3.7' in the same environment before running "
            "romoya_lebai_* robots or teleoperators."
        ) from exc


def get_field(data: Any, name: str) -> Any:
    if hasattr(data, name):
        return getattr(data, name)
    if isinstance(data, Mapping):
        return data[name]
    try:
        return data[name]
    except Exception as exc:
        raise AttributeError(f"Could not read field '{name}' from {type(data)}") from exc


def pose_to_dict(pose: Any) -> dict[str, float]:
    if isinstance(pose, Mapping):
        return {key: float(pose[key]) for key in TCP_KEYS}
    return {key: float(getattr(pose, key)) for key in TCP_KEYS}


def prefixed_pose_to_dict(prefix: str, pose: Any) -> dict[str, float]:
    return {f"{prefix}.{key}": value for key, value in pose_to_dict(pose).items()}


def get_float_sequence(data: Any, name: str, *, expected_len: int | None = None) -> list[float]:
    values = [float(v) for v in get_field(data, name)]
    if expected_len is not None and len(values) < expected_len:
        raise ValueError(f"Expected '{name}' to contain at least {expected_len} values, got {len(values)}")
    return values


def maybe_read_camera(camera: Any) -> Any:
    if hasattr(camera, "read_latest"):
        return camera.read_latest()
    if hasattr(camera, "async_read"):
        return camera.async_read()
    raise AttributeError(f"Unsupported camera interface: {type(camera)}")


def connect_arm(ip: str, *, port: int | None, simu: bool) -> Any:
    lebai_sdk = get_lebai_sdk()
    lebai_sdk.init()
    connect = lebai_sdk.connect

    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    if port is not None:
        attempts.append(((ip, port), {}))
        attempts.append(((ip,), {"port": port}))
    attempts.append(((ip,), {"simu": simu}))
    attempts.append(((ip,), {}))

    last_error = None
    for args, kwargs in attempts:
        try:
            return connect(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue

    raise TypeError(f"Unable to connect to Lebai arm with supported signatures: {last_error}")


def get_claw_data(arm: Any) -> dict[str, float | bool]:
    getter = getattr(arm, "get_claw_data", None)
    if callable(getter):
        data = getter()
        return {
            "force": float(get_field(data, "force")),
            "amplitude": float(get_field(data, "amplitude")),
            "hold_on": bool(get_field(data, "hold_on")),
        }

    legacy_getter = getattr(arm, "get_claw", None)
    if callable(legacy_getter):
        data = legacy_getter()
        if isinstance(data, Mapping):
            force = float(data.get("force", 0.0))
            amplitude = float(data.get("amplitude", 0.0))
            hold_on = bool(data.get("hold_on", False))
            return {"force": force, "amplitude": amplitude, "hold_on": hold_on}

        if hasattr(data, "force") or hasattr(data, "amplitude") or hasattr(data, "hold_on"):
            force = float(getattr(data, "force", 0.0))
            amplitude = float(getattr(data, "amplitude", 0.0))
            hold_on = bool(getattr(data, "hold_on", False))
            return {"force": force, "amplitude": amplitude, "hold_on": hold_on}

        if isinstance(data, (list, tuple)):
            values = list(data)
            force = float(values[0]) if len(values) >= 1 else 0.0
            amplitude = float(values[1]) if len(values) >= 2 else 0.0
            hold_on = bool(values[2]) if len(values) >= 3 else False
            return {"force": force, "amplitude": amplitude, "hold_on": hold_on}

    return {"force": 0.0, "amplitude": 0.0, "hold_on": False}


_KINEMATICS_CLIENT: Any | None = None


def _get_kinematics_client() -> Any:
    global _KINEMATICS_CLIENT

    sdk = get_lebai_sdk()
    if hasattr(sdk, "kinematics_forward") and hasattr(sdk, "kinematics_inverse"):
        init = getattr(sdk, "init", None)
        if callable(init):
            init()
        return sdk

    if _KINEMATICS_CLIENT is None:
        ip = os.environ.get("LEBAI_KINEMATICS_IP")
        if not ip:
            raise RuntimeError(
                "lebai_sdk does not expose generic kinematics helpers in this environment. "
                "Set LEBAI_KINEMATICS_IP so Romoya can open a cached SDK client for FK/IK."
            )
        port_str = os.environ.get("LEBAI_KINEMATICS_PORT")
        simu = os.environ.get("LEBAI_KINEMATICS_SIMU", "").lower() in {"1", "true", "yes"}
        port = int(port_str) if port_str else None
        _KINEMATICS_CLIENT = connect_arm(ip, port=port, simu=simu)

    return _KINEMATICS_CLIENT


def kinematics_forward(joint_positions: list[float]) -> dict[str, float]:
    client = _get_kinematics_client()
    result = client.kinematics_forward(list(joint_positions))
    return pose_to_dict(result)


def kinematics_inverse(tcp_pose: dict[str, float], seed_joints: list[float] | None = None) -> list[float]:
    client = _get_kinematics_client()
    target_pose = {key: float(tcp_pose[key]) for key in TCP_KEYS}
    if seed_joints is None:
        result = client.kinematics_inverse(target_pose)
    else:
        result = client.kinematics_inverse(target_pose, list(seed_joints))
    if hasattr(result, "joint_positions"):
        return [float(v) for v in result.joint_positions]
    return [float(v) for v in result]

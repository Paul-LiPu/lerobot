from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..lebai_sdk_utils import kinematics_inverse

DEFAULT_ROMOYA_OBS_STATE_NAMES = [
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
    "joint1.effort",
    "joint2.effort",
    "joint3.effort",
    "joint4.effort",
    "joint5.effort",
    "joint6.effort",
    "joint1.vel",
    "joint2.vel",
    "joint3.vel",
    "joint4.vel",
    "joint5.vel",
    "joint6.vel",
    "joint1.acc",
    "joint2.acc",
    "joint3.acc",
    "joint4.acc",
    "joint5.acc",
    "joint6.acc",
    "tcp.x",
    "tcp.y",
    "tcp.z",
    "tcp.rx",
    "tcp.ry",
    "tcp.rz",
]

DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES = [
    *DEFAULT_ROMOYA_OBS_STATE_NAMES,
    "joint1.temp",
    "joint2.temp",
    "joint3.temp",
    "joint4.temp",
    "joint5.temp",
    "joint6.temp",
    "joint1.voltage",
    "joint2.voltage",
    "joint3.voltage",
    "joint4.voltage",
    "joint5.voltage",
    "joint6.voltage",
    "flange_voltage",
    "flange.x",
    "flange.y",
    "flange.z",
    "flange.rx",
    "flange.ry",
    "flange.rz",
    "target_joint1.pos",
    "target_joint2.pos",
    "target_joint3.pos",
    "target_joint4.pos",
    "target_joint5.pos",
    "target_joint6.pos",
    "target_joint1.vel",
    "target_joint2.vel",
    "target_joint3.vel",
    "target_joint4.vel",
    "target_joint5.vel",
    "target_joint6.vel",
    "target_joint1.acc",
    "target_joint2.acc",
    "target_joint3.acc",
    "target_joint4.acc",
    "target_joint5.acc",
    "target_joint6.acc",
    "target_joint1.effort",
    "target_joint2.effort",
    "target_joint3.effort",
    "target_joint4.effort",
    "target_joint5.effort",
    "target_joint6.effort",
    "target_tcp.x",
    "target_tcp.y",
    "target_tcp.z",
    "target_tcp.rx",
    "target_tcp.ry",
    "target_tcp.rz",
]

DEFAULT_ROMOYA_ACTION_NAMES = [
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

DEFAULT_ROMOYA_TCP_ACTION_NAMES = [
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
    "joint1.effort",
    "joint2.effort",
    "joint3.effort",
    "joint4.effort",
    "joint5.effort",
    "joint6.effort",
    "joint1.vel",
    "joint2.vel",
    "joint3.vel",
    "joint4.vel",
    "joint5.vel",
    "joint6.vel",
    "joint1.acc",
    "joint2.acc",
    "joint3.acc",
    "joint4.acc",
    "joint5.acc",
    "joint6.acc",
    "tcp.x",
    "tcp.y",
    "tcp.z",
    "tcp.rx",
    "tcp.ry",
    "tcp.rz",
]

DEFAULT_ROMOYA_TCP_APPENDED_ACTION_NAMES = [
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
    "tcp.x",
    "tcp.y",
    "tcp.z",
    "tcp.rx",
    "tcp.ry",
    "tcp.rz",
]

DELTA_ACTION_MODE = "delta_joint_gripper_do1"
ABSOLUTE_ACTION_MODE = "absolute_joint_gripper_do1"
ABSOLUTE_TCP_ACTION_MODE = "absolute_tcp_gripper_do1"
DELTA_TCP_ACTION_MODE = "delta_tcp_gripper_do1"

JOINT_ACTION_NAMES = [
    "joint1.pos",
    "joint2.pos",
    "joint3.pos",
    "joint4.pos",
    "joint5.pos",
    "joint6.pos",
]
TCP_ACTION_NAMES = ["tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz"]
DEFAULT_STATE_FEATURE_NAMES = [*JOINT_ACTION_NAMES, "gripper.pos", "DO_1"]
JOINT_CONTROL = "joint_control"
TCP_CONTROL = "tcp_control"

BinaryStateSpec = float | None
BinaryActionSpec = tuple[float, float, float] | None


@dataclass(frozen=True)
class RomoyaTransformSpec:
    state_feature_names: list[str]
    action_feature_names: list[str]
    binary_state: list[BinaryStateSpec]
    binary_action: list[BinaryActionSpec]
    delta_action: list[bool]
    control_schema: str


def _binary_state_defaults(names: list[str]) -> list[BinaryStateSpec]:
    return [0.5 if name == "DO_1" else None for name in names]


def _binary_action_defaults(names: list[str]) -> list[BinaryActionSpec]:
    return [(0.5, 0.0, 1.0) if name == "DO_1" else None for name in names]


def transformed_action_names(action_feature_names: list[str], delta_action: list[bool]) -> list[str]:
    names = []
    for name, is_delta in zip(action_feature_names, delta_action, strict=True):
        names.append(f"delta_{name}" if is_delta else name)
    return names


def classify_control_schema(action_feature_names: list[str]) -> str:
    action_set = set(action_feature_names)
    optional_suffixes = {"gripper.pos", "DO_1"}
    if action_set.issubset({*JOINT_ACTION_NAMES, *optional_suffixes}) and set(JOINT_ACTION_NAMES).issubset(action_set):
        return JOINT_CONTROL
    if action_set.issubset({*TCP_ACTION_NAMES, *optional_suffixes}) and set(TCP_ACTION_NAMES).issubset(action_set):
        return TCP_CONTROL
    raise ValueError(
        "act_romoya action_feature_names must contain either all 6 joint pose dims or all 6 TCP pose dims. "
        "gripper.pos and DO_1 are optional extras."
    )


def validate_transform_spec(
    state_feature_names: list[str],
    action_feature_names: list[str],
    binary_state: list[BinaryStateSpec],
    binary_action: list[BinaryActionSpec],
    delta_action: list[bool],
    raw_observation_state_feature_names: list[str],
    raw_action_feature_names: list[str],
) -> str:
    if len(binary_state) != len(state_feature_names):
        raise ValueError("binary_state must have the same length as state_feature_names.")
    if len(binary_action) != len(action_feature_names):
        raise ValueError("binary_action must have the same length as action_feature_names.")
    if len(delta_action) != len(action_feature_names):
        raise ValueError("delta_action must have the same length as action_feature_names.")

    raw_state_name_set = set(raw_observation_state_feature_names)
    raw_action_name_set = set(raw_action_feature_names)
    for name in state_feature_names:
        if name not in raw_state_name_set:
            raise ValueError(f"State feature '{name}' is missing from raw observation.state schema.")
    for idx, name in enumerate(action_feature_names):
        if name not in raw_action_name_set:
            raise ValueError(f"Action feature '{name}' is missing from raw action schema.")
        if delta_action[idx] and name not in raw_state_name_set:
            raise ValueError(
                f"Delta action feature '{name}' requires a raw observation.state feature with the exact same name."
            )
        if delta_action[idx] and binary_action[idx] is not None:
            raise ValueError(f"Action feature '{name}' cannot be both binary and delta.")
        spec = binary_action[idx]
        if spec is not None and len(spec) != 3:
            raise ValueError(
                f"Action feature '{name}' binary_action spec must be a tuple of (threshold, low, high)."
            )
    return classify_control_schema(action_feature_names)


def legacy_transform_spec(
    action_mode: str,
    *,
    state_feature_names_to_keep: list[str] | None = None,
    joint_action_names: list[str] | None = None,
    gripper_action_name: str = "gripper.pos",
    do_action_names: list[str] | None = None,
) -> RomoyaTransformSpec:
    state_names = list(state_feature_names_to_keep or DEFAULT_STATE_FEATURE_NAMES)
    joints = list(joint_action_names or JOINT_ACTION_NAMES)
    do_names = list(do_action_names or ["DO_1"])
    if do_names != ["DO_1"]:
        raise ValueError("Legacy act_romoya compatibility only supports DO_1 as the modeled digital output.")

    if action_mode == DELTA_ACTION_MODE:
        action_names = [*joints, gripper_action_name, *do_names]
        delta_action = [True] * (len(joints) + 1) + [False] * len(do_names)
    elif action_mode == ABSOLUTE_ACTION_MODE:
        action_names = [*joints, gripper_action_name, *do_names]
        delta_action = [False] * len(action_names)
    elif action_mode == ABSOLUTE_TCP_ACTION_MODE:
        action_names = [*TCP_ACTION_NAMES, gripper_action_name, *do_names]
        delta_action = [False] * len(action_names)
    elif action_mode == DELTA_TCP_ACTION_MODE:
        action_names = [*TCP_ACTION_NAMES, gripper_action_name, *do_names]
        delta_action = [True] * (len(TCP_ACTION_NAMES) + 1) + [False] * len(do_names)
    else:
        raise ValueError(f"Unsupported action_mode: {action_mode}")

    return RomoyaTransformSpec(
        state_feature_names=state_names,
        action_feature_names=action_names,
        binary_state=_binary_state_defaults(state_names),
        binary_action=_binary_action_defaults(action_names),
        delta_action=delta_action,
        control_schema=classify_control_schema(action_names),
    )


def _index_map(names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(names)}


def _apply_state_binarization(values: Tensor, binary_state: list[BinaryStateSpec]) -> Tensor:
    output = values.clone()
    for idx, threshold in enumerate(binary_state):
        if threshold is not None:
            output[..., idx] = (output[..., idx] >= threshold).to(dtype=output.dtype)
    return output


def select_and_transform_state(
    raw_state: Tensor,
    raw_observation_state_feature_names: list[str],
    state_feature_names: list[str],
    binary_state: list[BinaryStateSpec],
) -> Tensor:
    state_indices = [_index_map(raw_observation_state_feature_names)[name] for name in state_feature_names]
    selected = raw_state[..., state_indices]
    return _apply_state_binarization(selected, binary_state)


def select_and_transform_action(
    raw_action: Tensor,
    raw_state: Tensor,
    raw_action_feature_names: list[str],
    raw_observation_state_feature_names: list[str],
    action_feature_names: list[str],
    binary_action: list[BinaryActionSpec],
    delta_action: list[bool],
) -> Tensor:
    action_index = _index_map(raw_action_feature_names)
    state_index = _index_map(raw_observation_state_feature_names)
    aligned_state = raw_state
    if raw_action.ndim == raw_state.ndim + 1:
        aligned_state = raw_state.unsqueeze(-2)
    transformed_columns = []
    for name, binary_spec, is_delta in zip(
        action_feature_names, binary_action, delta_action, strict=True
    ):
        values = raw_action[..., action_index[name]]
        if binary_spec is not None:
            threshold, _, _ = binary_spec
            values = (values >= threshold).to(dtype=raw_action.dtype)
        if is_delta:
            values = values - aligned_state[..., state_index[name]]
        transformed_columns.append(values.unsqueeze(-1))
    return torch.cat(transformed_columns, dim=-1)


def reconstruct_robot_action(
    transformed_action: Tensor,
    raw_state: Tensor,
    spec: RomoyaTransformSpec,
    raw_observation_state_feature_names: list[str],
    gripper_min: float = 0.0,
    gripper_max: float = 100.0,
) -> Tensor:
    state_index = _index_map(raw_observation_state_feature_names)
    output_action_index = _index_map(DEFAULT_ROMOYA_ACTION_NAMES)

    flat_action = transformed_action.reshape(-1, transformed_action.shape[-1])
    flat_state = raw_state.reshape(-1, raw_state.shape[-1])
    reconstructed_values: dict[str, Tensor] = {}

    for column, (name, binary_spec, is_delta) in enumerate(
        zip(spec.action_feature_names, spec.binary_action, spec.delta_action, strict=True)
    ):
        values = flat_action[:, column]
        if binary_spec is not None:
            threshold, low, high = binary_spec
            values = torch.where(values >= threshold, torch.full_like(values, high), torch.full_like(values, low))
        if is_delta:
            values = values + flat_state[:, state_index[name]]
        reconstructed_values[name] = values

    reconstructed = torch.zeros(
        flat_action.shape[0],
        len(DEFAULT_ROMOYA_ACTION_NAMES),
        dtype=flat_action.dtype,
        device=flat_action.device,
    )
    if spec.control_schema == JOINT_CONTROL:
        for joint_name in JOINT_ACTION_NAMES:
            reconstructed[:, output_action_index[joint_name]] = reconstructed_values[joint_name]
    elif spec.control_schema == TCP_CONTROL:
        seed_joints = torch.stack([flat_state[:, state_index[name]] for name in JOINT_ACTION_NAMES], dim=-1)
        target_tcp = torch.stack([reconstructed_values[name] for name in TCP_ACTION_NAMES], dim=-1)
        joint_targets = []
        for tcp_row, seed_row in zip(target_tcp, seed_joints, strict=True):
            tcp_pose = {name.split(".", 1)[1]: float(value) for name, value in zip(TCP_ACTION_NAMES, tcp_row, strict=True)}
            joint_targets.append(kinematics_inverse(tcp_pose, [float(v) for v in seed_row]))
        reconstructed[:, [output_action_index[name] for name in JOINT_ACTION_NAMES]] = torch.tensor(
            joint_targets, dtype=flat_action.dtype, device=flat_action.device
        )
    else:
        raise ValueError(f"Unsupported control schema: {spec.control_schema}")

    if "gripper.pos" in reconstructed_values:
        gripper_target = reconstructed_values["gripper.pos"]
    else:
        gripper_target = flat_state[:, state_index["gripper.pos"]]
    reconstructed[:, output_action_index["gripper.pos"]] = torch.clamp(
        gripper_target,
        min=gripper_min,
        max=gripper_max,
    )
    reconstructed[:, output_action_index["gripper.force"]] = flat_state[:, state_index["gripper.force"]]
    if "DO_1" in reconstructed_values:
        reconstructed[:, output_action_index["DO_0"]] = reconstructed_values["DO_1"]
        reconstructed[:, output_action_index["DO_1"]] = reconstructed_values["DO_1"]
    else:
        reconstructed[:, output_action_index["DO_0"]] = flat_state[:, state_index["DO_0"]]
        reconstructed[:, output_action_index["DO_1"]] = flat_state[:, state_index["DO_1"]]
    return reconstructed.reshape(*transformed_action.shape[:-1], len(DEFAULT_ROMOYA_ACTION_NAMES))


def tensor_to_python_lists(values: Any) -> Any:
    if isinstance(values, Tensor):
        return values.detach().cpu().tolist()
    if hasattr(values, "tolist"):
        return values.tolist()
    return values

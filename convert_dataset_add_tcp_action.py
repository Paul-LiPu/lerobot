#!/usr/bin/env python3
"""Create a new dataset with tcp.* appended to action using Lebai forward kinematics."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot_robot_romoya.lebai_sdk_utils import TCP_KEYS, kinematics_forward


JOINT_ACTION_NAMES = [f"joint{i}.pos" for i in range(1, 7)]
REQUIRED_ACTION_NAMES = [*JOINT_ACTION_NAMES, "gripper.pos", "gripper.force", "DO_0", "DO_1"]
TCP_ACTION_NAMES = [f"tcp.{key}" for key in TCP_KEYS]
METADATA_KEYS_TO_DROP = {"index", "episode_index", "frame_index", "task_index"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Source dataset repo id")
    parser.add_argument("--root", type=Path, default=None, help="Optional source dataset root")
    parser.add_argument("--new-repo-id", default=None, help="Destination dataset repo id")
    parser.add_argument("--new-root", type=Path, default=None, help="Optional destination dataset root")
    parser.add_argument(
        "--overwrite-tcp",
        action="store_true",
        help="If source action already contains tcp.*, recompute and replace them in the destination action",
    )
    return parser.parse_args()


def _to_action_dict(action_values, action_names: list[str]) -> dict[str, float]:
    return {name: float(action_values[i]) for i, name in enumerate(action_names)}


def _build_output_action_names(action_names: list[str], overwrite_tcp: bool) -> list[str]:
    present_tcp = [name for name in TCP_ACTION_NAMES if name in action_names]
    if present_tcp and not overwrite_tcp:
        raise ValueError(
            f"Source dataset action already contains tcp fields: {present_tcp}. "
            "Use --overwrite-tcp to recompute and replace them."
        )

    missing_required = [name for name in REQUIRED_ACTION_NAMES if name not in action_names]
    if missing_required:
        raise ValueError(f"Source dataset action is missing required names: {missing_required}")

    base_action_names = [name for name in action_names if name not in TCP_ACTION_NAMES]
    return [*base_action_names, *TCP_ACTION_NAMES]


def _build_new_action(action_values, action_names: list[str], output_action_names: list[str]) -> np.ndarray:
    action_dict = _to_action_dict(action_values, action_names)
    joint_positions = [action_dict[name] for name in JOINT_ACTION_NAMES]
    tcp_pose = kinematics_forward(joint_positions)

    values = []
    for name in output_action_names:
        if name in TCP_ACTION_NAMES:
            values.append(float(tcp_pose[name.split(".", 1)[1]]))
        else:
            values.append(float(action_dict[name]))
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    args = parse_args()

    src = LeRobotDataset(args.repo_id, root=args.root)
    src_action_feature = copy.deepcopy(src.meta.features[ACTION])
    src_action_names = list(src_action_feature["names"])

    output_action_names = _build_output_action_names(src_action_names, args.overwrite_tcp)
    new_repo_id = args.new_repo_id or f"{args.repo_id}_tcp_action"

    new_features = copy.deepcopy(src.meta.features)
    new_features[ACTION]["shape"] = [len(output_action_names)]
    new_features[ACTION]["names"] = output_action_names

    dst = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=src.meta.fps,
        root=args.new_root,
        robot_type=src.meta.robot_type,
        features=new_features,
        use_videos=len(src.meta.video_keys) > 0,
    )

    current_episode = None
    for idx in tqdm(range(len(src)), desc="Converting", unit="frame"):
        item = src[idx]
        episode_index = int(item["episode_index"])
        if current_episode is None:
            current_episode = episode_index
        elif episode_index != current_episode:
            dst.save_episode()
            current_episode = episode_index

        frame = {}
        for key, value in item.items():
            if key == "task" or key in METADATA_KEYS_TO_DROP:
                continue
            if key == ACTION:
                frame[ACTION] = _build_new_action(value, src_action_names, output_action_names)
            else:
                frame[key] = value
        frame["task"] = item["task"]
        dst.add_frame(frame)

    if dst.episode_buffer is not None and dst.episode_buffer["size"] > 0:
        dst.save_episode()
    dst.finalize()

    print(f"Created dataset: {new_repo_id}")
    print(f"Action names: {output_action_names}")


if __name__ == "__main__":
    main()

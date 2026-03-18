#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STATE

RESET_STATE_KEYS = [*(f"joint{i}.pos" for i in range(1, 7)), "gripper.pos", "DO_0", "DO_1"]


def _validate_reset_state(saved_reset_state: dict[str, float]) -> dict[str, float]:
    missing = [key for key in RESET_STATE_KEYS if key not in saved_reset_state]
    if missing:
        raise ValueError(f"Initial pose is missing keys: {missing}")
    return {key: float(saved_reset_state[key]) for key in RESET_STATE_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract an initial pose JSON from a dataset episode.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-root")
    args = parser.parse_args()

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=[args.episode])
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == args.episode)
    if len(episode_frames) == 0:
        raise ValueError(f"Episode {args.episode} not found in dataset {args.dataset_repo_id}")

    state_names = dataset.features[OBS_STATE]["names"]
    state_values = episode_frames[0][OBS_STATE]
    if len(state_names) != len(state_values):
        raise ValueError("Observation state names length does not match state values length")

    state = dict(zip(state_names, state_values, strict=True))
    payload = _validate_reset_state(state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote initial pose to {output_path}")


if __name__ == "__main__":
    main()

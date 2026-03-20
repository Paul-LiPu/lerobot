#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_robot_romoya.policies.romoya_transforms import (
    DEFAULT_ROMOYA_ACTION_NAMES,
    DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES,
)


def _load_schema_from_dataset(repo_id: str, root: str | None) -> tuple[list[str], list[str]]:
    dataset = LeRobotDataset(repo_id, root=root, download_videos=False)
    raw_state = list(dataset.meta.features[OBS_STATE].get("names", []))
    raw_action = list(dataset.meta.features[ACTION].get("names", []))
    if not raw_state or not raw_action:
        raise ValueError(f"Dataset {repo_id} is missing {OBS_STATE} or {ACTION} feature names.")
    return raw_state, raw_action


def _patch_top_level_config(data: dict, raw_state: list[str], raw_action: list[str]) -> dict:
    updated = dict(data)
    updated["raw_observation_state_feature_names"] = list(raw_state)
    updated["raw_action_feature_names"] = list(raw_action)
    return updated


def _patch_processor_config(data: dict, raw_state: list[str], raw_action: list[str]) -> dict:
    updated = dict(data)
    updated["steps"] = list(updated.get("steps", []))
    for step in updated["steps"]:
        config = step.get("config")
        if not isinstance(config, dict):
            continue
        if step.get("registry_name") in {"act_romoya_preprocess_v1", "act_romoya_postprocess_v1"}:
            config["raw_observation_state_feature_names"] = list(raw_state)
            config["raw_action_feature_names"] = list(raw_action)
    return updated


def _patch_model_dir(model_dir: Path, raw_state: list[str], raw_action: list[str]) -> None:
    config_path = model_dir / "config.json"
    pre_path = model_dir / "policy_preprocessor.json"
    post_path = model_dir / "policy_postprocessor.json"

    config_data = json.loads(config_path.read_text())
    pre_data = json.loads(pre_path.read_text())
    post_data = json.loads(post_path.read_text())

    config_path.write_text(json.dumps(_patch_top_level_config(config_data, raw_state, raw_action), indent=2) + "\n")
    pre_path.write_text(json.dumps(_patch_processor_config(pre_data, raw_state, raw_action), indent=2) + "\n")
    post_path.write_text(json.dumps(_patch_processor_config(post_data, raw_state, raw_action), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair saved act_romoya checkpoints with correct raw schemas.")
    parser.add_argument("--model-dir", required=True, help="Directory containing config.json and policy_*processor.json")
    parser.add_argument(
        "--output-dir",
        help="Optional destination directory. If set, copy model-dir there first and patch the copy.",
    )
    parser.add_argument("--src-repo-id", help="Optional dataset repo id to read raw state/action schemas from.")
    parser.add_argument("--src-root", help="Optional dataset root for --src-repo-id.")
    parser.add_argument(
        "--use-default-lebai-schemas",
        action="store_true",
        help="Use the current wide Lebai observation schema and 10D action schema.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    target_dir = model_dir
    if args.output_dir:
        target_dir = Path(args.output_dir)
        if target_dir.exists():
            raise FileExistsError(f"Output directory already exists: {target_dir}")
        shutil.copytree(model_dir, target_dir)

    if args.src_repo_id:
        raw_state, raw_action = _load_schema_from_dataset(args.src_repo_id, args.src_root)
    else:
        raw_state = list(DEFAULT_ROMOYA_WIDE_OBS_STATE_NAMES)
        raw_action = list(DEFAULT_ROMOYA_ACTION_NAMES)

    if not args.src_repo_id and not args.use_default_lebai_schemas:
        print("No source dataset provided; using current default Lebai raw schemas.")

    _patch_model_dir(target_dir, raw_state, raw_action)
    print(f"Patched Romoya checkpoint schema under: {target_dir}")


if __name__ == "__main__":
    main()

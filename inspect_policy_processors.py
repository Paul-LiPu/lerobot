#!/usr/bin/env python3
"""Inspect saved LeRobot policy processor configs and normalization stats.

Usage:
  python3 inspect_policy_processors.py --model-dir /path/to/pretrained_model
  python3 inspect_policy_processors.py \
    --pre-json /path/policy_preprocessor.json \
    --post-json /path/policy_postprocessor.json \
    --pre-stats /path/policy_preprocessor_step_4_normalizer_processor.safetensors \
    --post-stats /path/policy_postprocessor_step_0_unnormalizer_processor.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_safetensors(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base = 8 + header_len

        tensors: dict[str, dict[str, Any]] = {}
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            start, end = spec["data_offsets"]
            dtype = spec["dtype"]
            shape = spec["shape"]
            f.seek(base + start)
            raw = f.read(end - start)
            if dtype == "F32":
                values = list(struct.unpack("<" + "f" * ((end - start) // 4), raw))
            elif dtype == "I64":
                values = list(struct.unpack("<" + "q" * ((end - start) // 8), raw))
            else:
                values = []
            tensors[name] = {"dtype": dtype, "shape": shape, "values": values}
        return tensors


def find_step(steps: list[dict[str, Any]], registry_name: str) -> dict[str, Any] | None:
    for step in steps:
        if step.get("registry_name") == registry_name:
            return step
    return None


def summarize_preprocessor(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    steps = config["steps"]
    preprocess = find_step(steps, "act_romoya_preprocess_v1")
    normalizer = find_step(steps, "normalizer_processor")
    if preprocess is None or normalizer is None:
        raise ValueError("Expected act_romoya_preprocess_v1 and normalizer_processor in preprocessor config.")

    pre_cfg = preprocess["config"]
    features = normalizer["config"]["features"]

    state_names = pre_cfg.get("state_feature_names") or pre_cfg["state_feature_names_to_keep"]
    if "action_feature_names" in pre_cfg:
        action_names = list(pre_cfg["action_feature_names"])
    else:
        action_names = list(pre_cfg["joint_action_names"]) + [pre_cfg["gripper_action_name"]] + list(
            pre_cfg["do_action_names"]
        )

    print("Preprocessor")
    print(f"  observation.state shape: {features['observation.state']['shape']}")
    print(f"  action shape: {features['action']['shape']}")
    print(f"  state names: {state_names}")
    print(f"  transformed action names: {action_names}")
    print("  image keys:")
    for key in features:
        if key.startswith("observation.images."):
            print(f"    - {key}: {features[key]['shape']}")
    print("")

    return state_names, action_names


def summarize_postprocessor(config: dict[str, Any]) -> list[str]:
    steps = config["steps"]
    postprocess = find_step(steps, "act_romoya_postprocess_v1")
    unnormalizer = find_step(steps, "unnormalizer_processor")
    if postprocess is None or unnormalizer is None:
        raise ValueError("Expected act_romoya_postprocess_v1 and unnormalizer_processor in postprocessor config.")

    post_cfg = postprocess["config"]
    if "action_feature_names" in post_cfg:
        action_names = list(post_cfg["action_feature_names"])
    else:
        action_names = list(post_cfg["joint_action_names"]) + [post_cfg["gripper_action_name"]] + list(
            post_cfg["do_action_names"]
        )

    print("Postprocessor")
    print(f"  unnormalized action shape: {unnormalizer['config']['features']['action']['shape']}")
    print(f"  transformed action names: {action_names}")
    print(f"  do_threshold: {post_cfg['do_threshold']}")
    print(f"  sigmoid_do_outputs: {post_cfg['sigmoid_do_outputs']}")
    print(f"  gripper range: [{post_cfg['gripper_min']}, {post_cfg['gripper_max']}]")
    print("")

    return action_names


def print_vector_stat(
    prefix: str,
    names: list[str],
    stats: dict[str, dict[str, Any]],
) -> None:
    means = stats.get(f"{prefix}.mean", {}).get("values", [])
    stds = stats.get(f"{prefix}.std", {}).get("values", [])
    mins = stats.get(f"{prefix}.min", {}).get("values", [])
    maxs = stats.get(f"{prefix}.max", {}).get("values", [])

    if not means or not stds:
        print(f"No {prefix} stats found.")
        print("")
        return

    print(f"{prefix} stats")
    for i, name in enumerate(names):
        mean = means[i] if i < len(means) else None
        std = stds[i] if i < len(stds) else None
        min_val = mins[i] if i < len(mins) else None
        max_val = maxs[i] if i < len(maxs) else None
        flags = []
        if std == 0.0:
            flags.append("ZERO_STD")
        if min_val is not None and max_val is not None and min_val == max_val:
            flags.append("CONSTANT")
        flag_text = f" [{' '.join(flags)}]" if flags else ""
        print(
            f"  {i}: {name}: mean={mean:.6f}, std={std:.6f}, min={min_val:.6f}, max={max_val:.6f}{flag_text}"
        )
    print("")


def build_default_paths(model_dir: Path) -> dict[str, Path]:
    return {
        "pre_json": model_dir / "policy_preprocessor.json",
        "post_json": model_dir / "policy_postprocessor.json",
        "pre_stats": model_dir / "policy_preprocessor_step_4_normalizer_processor.safetensors",
        "post_stats": model_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--pre-json", type=Path)
    parser.add_argument("--post-json", type=Path)
    parser.add_argument("--pre-stats", type=Path)
    parser.add_argument("--post-stats", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model_dir is not None:
        paths = build_default_paths(args.model_dir)
    else:
        paths = {
            "pre_json": args.pre_json,
            "post_json": args.post_json,
            "pre_stats": args.pre_stats,
            "post_stats": args.post_stats,
        }

    missing = [name for name, path in paths.items() if path is None or not Path(path).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")

    pre_json = load_json(Path(paths["pre_json"]))
    post_json = load_json(Path(paths["post_json"]))
    pre_stats = load_safetensors(Path(paths["pre_stats"]))
    post_stats = load_safetensors(Path(paths["post_stats"]))

    state_names, action_names = summarize_preprocessor(pre_json)
    summarize_postprocessor(post_json)

    print_vector_stat("observation.state", state_names, pre_stats)
    print_vector_stat("action", action_names, pre_stats)
    print("Note: postprocessor unnormalizer stats should match the transformed action stats above.")
    print_vector_stat("action", action_names, post_stats)


if __name__ == "__main__":
    main()

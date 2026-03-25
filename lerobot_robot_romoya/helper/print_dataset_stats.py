#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def print_vector_stats(features: dict, stats: dict, key: str) -> None:
    feature = features.get(key)
    stat = stats.get(key)
    if feature is None or stat is None:
        print(f"{key}: not found")
        return
    names = feature.get("names") or [f"{key}[{i}]" for i in range(len(stat.get("mean", [])))]
    print(f"{key} stats:")
    for i, name in enumerate(names):
        mean = stat["mean"][i]
        std = stat["std"][i]
        min_val = stat["min"][i]
        max_val = stat["max"][i]
        print(f"  {i}: {name}: mean={mean:.6f}, std={std:.6f}, min={min_val:.6f}, max={max_val:.6f}")


def main() -> None:
    dataset_repo_id = sys.argv[1]
    dataset_root_override = sys.argv[2] or None
    source_repo_ids = sys.argv[3:]
    if dataset_root_override:
        dataset_root = Path(dataset_root_override)
    else:
        hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        hf_lerobot_home = Path(os.getenv("HF_LEROBOT_HOME", str(hf_home / "lerobot")))
        dataset_root = hf_lerobot_home / dataset_repo_id
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"

    if not info_path.is_file() or not stats_path.is_file():
        print(f"Dataset stats not found locally under {dataset_root}; skipping stats print.")
        return

    info = json.loads(info_path.read_text())
    stats = json.loads(stats_path.read_text())
    features = info.get("features", {})

    print(f"Training dataset: {dataset_repo_id}")
    print(f"Dataset root: {dataset_root}")
    if source_repo_ids:
        print("Source datasets:")
        for repo_id in source_repo_ids:
            print(f"  - {repo_id}")
    romoya_prepare = info.get("romoya_prepare")
    if romoya_prepare is not None:
        print(f"Prepared dataset source: {romoya_prepare.get('source_repo_id')}")
    print_vector_stats(features, stats, "observation.state")
    print_vector_stats(features, stats, "action")


if __name__ == "__main__":
    main()

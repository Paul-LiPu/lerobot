#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    config_path = Path(sys.argv[1])
    src_repo_id = sys.argv[2]
    dst_repo_id = sys.argv[3]
    dst_root = sys.argv[4] or None

    payload = json.loads(config_path.read_text())
    policy = payload.get("policy", payload)
    expected = {
        "action_mode": policy.get("action_mode"),
        "state_feature_names": policy.get("state_feature_names"),
        "action_feature_names": policy.get("action_feature_names"),
        "binary_state": policy.get("binary_state"),
        "binary_action": policy.get("binary_action"),
        "delta_action": policy.get("delta_action"),
        "resize_shape": (
            (((payload.get("dataset") or {}).get("image_transforms") or {}).get("pre_tfs") or {})
            .get("resize", {})
            .get("kwargs", {})
            .get("size")
        ),
    }

    if dst_root:
        base_root = Path(dst_root)
    else:
        hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        hf_lerobot_home = Path(os.getenv("HF_LEROBOT_HOME", str(hf_home / "lerobot")))
        base_root = hf_lerobot_home / dst_repo_id
    info_path = base_root / "meta" / "info.json"
    stats_path = base_root / "meta" / "stats.json"
    if not info_path.is_file() or not stats_path.is_file():
        print("missing")
        return

    info = json.loads(info_path.read_text())
    romoya_prepare = info.get("romoya_prepare") or {}
    if romoya_prepare.get("source_repo_id") != src_repo_id:
        print("stale")
        return
    if not romoya_prepare.get("source_raw_observation_state_feature_names") or not romoya_prepare.get(
        "source_raw_action_feature_names"
    ):
        print("stale")
        return

    def eq(key: str) -> bool:
        return romoya_prepare.get(key) == expected.get(key)

    if expected["action_mode"] is not None:
        is_match = eq("action_mode")
    else:
        is_match = all(
            eq(key)
            for key in (
                "action_mode",
                "state_feature_names",
                "action_feature_names",
                "binary_state",
                "binary_action",
                "delta_action",
                "resize_shape",
            )
        )

    print("match" if is_match else "stale")


if __name__ == "__main__":
    main()

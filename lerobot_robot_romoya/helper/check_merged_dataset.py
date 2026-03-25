#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    merged_repo_id = sys.argv[1]
    merged_root_override = sys.argv[2] or None
    source_repo_ids = sys.argv[3:]

    if merged_root_override:
        dataset_root = Path(merged_root_override)
    else:
        hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        hf_lerobot_home = Path(os.getenv("HF_LEROBOT_HOME", str(hf_home / "lerobot")))
        dataset_root = hf_lerobot_home / merged_repo_id

    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"
    if not info_path.is_file() or not stats_path.is_file():
        print("missing")
        return

    info = json.loads(info_path.read_text())
    romoya_merge = info.get("romoya_merge") or {}
    if romoya_merge.get("source_repo_ids") == source_repo_ids:
        print("match")
    else:
        print("stale")


if __name__ == "__main__":
    main()

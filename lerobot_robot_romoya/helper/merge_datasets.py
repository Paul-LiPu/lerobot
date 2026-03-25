#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    merged_repo_id = sys.argv[1]
    merged_root_override = sys.argv[2] or None
    source_repo_ids = sys.argv[3:]

    datasets = [LeRobotDataset(repo_id) for repo_id in source_repo_ids]
    merged_dataset = merge_datasets(
        datasets,
        output_repo_id=merged_repo_id,
        output_dir=Path(merged_root_override) if merged_root_override else None,
    )
    merged_dataset.meta.info["romoya_merge"] = {
        "source_repo_ids": source_repo_ids,
    }
    write_info(merged_dataset.meta.info, merged_dataset.root)


if __name__ == "__main__":
    main()

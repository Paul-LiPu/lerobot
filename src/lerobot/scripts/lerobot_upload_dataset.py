#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


@dataclass
class UploadDatasetConfig:
    # Dataset identifier on the Hub, e.g. "user/my-dataset".
    repo_id: str
    # Local dataset root. If omitted, defaults to the LeRobot cache path for repo_id.
    root: str | Path | None = None
    # Optional Hub branch.
    branch: str | None = None
    # Whether to upload videos together with the dataset.
    push_videos: bool = True
    # Whether to create the Hub dataset as private.
    private: bool = False
    # Optional tags for the dataset card.
    tags: list[str] | None = None
    # Dataset card license field.
    license: str = "apache-2.0"
    # Whether to tag the upload with the current codebase version.
    tag_version: bool = True
    # Restrict uploaded files to these patterns.
    allow_patterns: list[str] | str | None = None
    # Use the large-folder uploader for big datasets.
    upload_large_folder: bool = False


@parser.wrap()
def upload_dataset(cfg: UploadDatasetConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    logging.info(
        "Uploading dataset %s from %s (%s episodes, %s frames)",
        cfg.repo_id,
        dataset.root,
        dataset.meta.total_episodes,
        dataset.meta.total_frames,
    )
    dataset.push_to_hub(
        branch=cfg.branch,
        tags=cfg.tags,
        license=cfg.license,
        tag_version=cfg.tag_version,
        push_videos=cfg.push_videos,
        private=cfg.private,
        allow_patterns=cfg.allow_patterns,
        upload_large_folder=cfg.upload_large_folder,
    )


def main() -> None:
    upload_dataset()


if __name__ == "__main__":
    main()

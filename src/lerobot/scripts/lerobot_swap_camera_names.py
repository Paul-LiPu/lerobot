#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import INFO_PATH, STATS_PATH, load_json, write_json
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


def _normalize_camera_key(name: str) -> str:
    return name if name.startswith("observation.images.") else f"observation.images.{name}"


def _swap_ordered_keys(mapping: dict, key_a: str, key_b: str) -> dict:
    swapped = {}
    for key, value in mapping.items():
        if key == key_a:
            swapped[key_b] = value
        elif key == key_b:
            swapped[key_a] = value
        else:
            swapped[key] = value
    return swapped


def _swap_columns(df: pd.DataFrame, key_a: str, key_b: str) -> pd.DataFrame:
    if key_a in df.columns and key_b in df.columns:
        tmp = df[key_a].copy()
        df[key_a] = df[key_b]
        df[key_b] = tmp
    elif key_a in df.columns:
        df = df.rename(columns={key_a: key_b})
    elif key_b in df.columns:
        df = df.rename(columns={key_b: key_a})
    return df


def _swap_prefixed_columns(df: pd.DataFrame, prefix_a: str, prefix_b: str) -> pd.DataFrame:
    rename_map = {}
    for column in df.columns:
        if column.startswith(prefix_a):
            rename_map[column] = column.replace(prefix_a, "__swap_tmp__", 1)
        elif column.startswith(prefix_b):
            rename_map[column] = column.replace(prefix_b, prefix_a, 1)
    if rename_map:
        df = df.rename(columns=rename_map)
        df = df.rename(columns={col: col.replace("__swap_tmp__", prefix_b, 1) for col in df.columns if "__swap_tmp__" in col})
    return df


def _prepare_output_path(repo_id: str, root: str | None, new_repo_id: str | None, new_root: str | None) -> tuple[str, Path, Path]:
    input_path = Path(root) if root else HF_LEROBOT_HOME / repo_id
    output_repo_id = new_repo_id if new_repo_id else f"{repo_id}_swapped"
    output_path = Path(new_root) if new_root else HF_LEROBOT_HOME / output_repo_id
    source_path = input_path

    if output_path == input_path and input_path.exists():
        backup_path = input_path.with_name(input_path.name + "_old")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.move(input_path, backup_path)
        source_path = backup_path

    return output_repo_id, source_path, output_path


def _swap_video_directories(root: Path, key_a: str, key_b: str) -> None:
    videos_dir = root / "videos"
    dir_a = videos_dir / key_a
    dir_b = videos_dir / key_b
    if not dir_a.exists() and not dir_b.exists():
        return

    tmp = videos_dir / "__swap_tmp__"
    if tmp.exists():
        shutil.rmtree(tmp)
    if dir_a.exists():
        dir_a.rename(tmp)
    if dir_b.exists():
        dir_b.rename(dir_a)
    if tmp.exists():
        tmp.rename(dir_b)


@dataclass
class SwapCameraNamesConfig:
    repo_id: str
    camera_a: str
    camera_b: str
    root: str | None = None
    new_repo_id: str | None = None
    new_root: str | None = None
    push_to_hub: bool = False


@parser.wrap()
def swap_camera_names(cfg: SwapCameraNamesConfig) -> None:
    init_logging()

    key_a = _normalize_camera_key(cfg.camera_a)
    key_b = _normalize_camera_key(cfg.camera_b)

    output_repo_id, source_root, output_root = _prepare_output_path(
        cfg.repo_id, cfg.root, cfg.new_repo_id, cfg.new_root
    )

    if output_root.exists():
        raise FileExistsError(f"Output dataset path already exists: {output_root}")

    logging.info("Copying dataset from %s to %s", source_root, output_root)
    shutil.copytree(source_root, output_root)

    info_path = output_root / INFO_PATH
    info = load_json(info_path)
    features = info["features"]
    if key_a not in features or key_b not in features:
        raise ValueError(
            f"Both camera keys must exist in dataset features. Missing: "
            f"{[key for key in (key_a, key_b) if key not in features]}"
        )
    info["features"] = _swap_ordered_keys(features, key_a, key_b)
    write_json(info, info_path)

    stats_path = output_root / STATS_PATH
    if stats_path.exists():
        stats = load_json(stats_path)
        if key_a in stats or key_b in stats:
            stats = _swap_ordered_keys(stats, key_a, key_b)
            write_json(stats, stats_path)

    for parquet_path in sorted((output_root / "data").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        df = _swap_columns(df, key_a, key_b)
        df.to_parquet(parquet_path, index=False)

    for parquet_path in sorted((output_root / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        df = _swap_prefixed_columns(df, f"videos/{key_a}/", f"videos/{key_b}/")
        df = _swap_prefixed_columns(df, f"stats/{key_a}/", f"stats/{key_b}/")
        df.to_parquet(parquet_path, index=False)

    _swap_video_directories(output_root, key_a, key_b)

    swapped_dataset = LeRobotDataset(output_repo_id, root=output_root)
    logging.info("Swapped camera names: %s <-> %s", key_a, key_b)
    logging.info(
        "Dataset saved to %s (%s episodes, %s frames)",
        output_root,
        swapped_dataset.meta.total_episodes,
        swapped_dataset.meta.total_frames,
    )

    if cfg.push_to_hub:
        logging.info("Pushing swapped dataset to hub as %s", output_repo_id)
        swapped_dataset.push_to_hub()


def main():
    swap_camera_names()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.io_utils import write_info, write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.utils.constants import ACTION, OBS_STATE


def _sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _resize_image_tensor(image: torch.Tensor, resize_shape: tuple[int, int]) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C, H, W), got {tuple(image.shape)}")
    input_dtype = image.dtype
    resized = F.interpolate(
        image.unsqueeze(0).to(dtype=torch.float32),
        size=resize_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    if input_dtype == torch.uint8:
        return resized.round().clamp(0, 255).to(dtype=torch.uint8)
    return resized.clamp(0.0, 1.0).to(dtype=input_dtype)


def _dataset_matches_video_target(
    dataset: LeRobotDataset,
    resize_shape: tuple[int, int],
    expected_video_codec: str,
) -> bool:
    target_h, target_w = resize_shape
    for key in dataset.meta.camera_keys:
        feature = dataset.meta.features.get(key, {})
        shape = tuple(feature.get("shape", ()))
        info = feature.get("info") or {}
        if len(shape) != 3 or shape[0] != target_h or shape[1] != target_w:
            return False
        if info.get("video.height") != target_h or info.get("video.width") != target_w:
            return False
        if info.get("video.codec") != expected_video_codec:
            return False
    return True


def _make_normalized_features(
    dataset: LeRobotDataset,
    resize_shape: tuple[int, int],
) -> dict[str, dict[str, Any]]:
    resize_h, resize_w = resize_shape
    features = copy.deepcopy(dataset.meta.features)
    for key in dataset.meta.camera_keys:
        feature = dict(features[key])
        channels = feature["shape"][-1]
        feature["shape"] = (resize_h, resize_w, channels)
        feature.pop("info", None)
        features[key] = feature
    return features


def _normalize_dataset(
    dataset: LeRobotDataset,
    dst_repo_id: str,
    dst_root: Path,
    resize_shape: tuple[int, int],
    vcodec: str,
) -> LeRobotDataset:
    if dst_root.exists():
        shutil.rmtree(dst_root)

    features = _make_normalized_features(dataset, resize_shape)
    out_dataset = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=dataset.meta.fps,
        features=features,
        robot_type=dataset.meta.info.get("robot_type"),
        root=dst_root,
        use_videos=bool(dataset.meta.video_keys),
        tolerance_s=dataset.tolerance_s,
        vcodec=vcodec,
    )

    camera_keys = list(dataset.meta.camera_keys)
    passthrough_keys = [key for key in dataset.features if key not in DEFAULT_FEATURES]

    dataset._ensure_hf_dataset_loaded()
    num_episodes = len(dataset.meta.episodes["dataset_from_index"])
    for episode_idx in range(num_episodes):
        start = int(dataset.meta.episodes["dataset_from_index"][episode_idx])
        end = int(dataset.meta.episodes["dataset_to_index"][episode_idx])
        for frame_idx in range(start, end):
            item = dataset[frame_idx]
            frame: dict[str, Any] = {"task": item["task"]}
            for key in passthrough_keys:
                if key in {"task_index", "frame_index", "episode_index", "index"}:
                    continue
                value = item[key]
                if key in camera_keys:
                    resized_image = _resize_image_tensor(value, resize_shape)
                    frame[key] = resized_image.permute(1, 2, 0).contiguous().cpu().numpy()
                else:
                    frame[key] = value
            out_dataset.add_frame(frame)
        out_dataset.save_episode()

    out_dataset.finalize()
    out_dataset.meta.info["romoya_normalize"] = {
        "source_repo_id": dataset.repo_id,
        "resize_shape": list(resize_shape),
        "vcodec": vcodec,
    }
    write_info(out_dataset.meta.info, out_dataset.root)
    return LeRobotDataset(repo_id=dst_repo_id, root=dst_root)


def _recompute_state_action_stats(dataset: LeRobotDataset) -> None:
    """Recompute observation.state and action stats from merged parquet data.

    This avoids inheriting stale stats from source datasets when the merged dataset
    features are already transformed to the Romoya schema.
    """
    parquet_files = sorted((dataset.root / "data").glob("*/*.parquet"))
    if not parquet_files:
        return

    state_feature = dataset.meta.features.get(OBS_STATE, {})
    action_feature = dataset.meta.features.get(ACTION, {})
    state_shape = tuple(state_feature.get("shape", ()))
    action_shape = tuple(action_feature.get("shape", ()))
    state_names = list(state_feature.get("names", []))
    action_names = list(action_feature.get("names", []))

    stats_list: list[dict[str, dict[str, Any]]] = []
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path).reset_index(drop=True)
        episode_stats = compute_episode_stats(
            {
                OBS_STATE: df[OBS_STATE].tolist(),
                ACTION: df[ACTION].tolist(),
            },
            {
                OBS_STATE: {"dtype": "float32", "shape": state_shape, "names": state_names},
                ACTION: {"dtype": "float32", "shape": action_shape, "names": action_names},
            },
        )
        stats_list.append(episode_stats)

    aggregated = aggregate_stats(stats_list)
    new_stats = copy.deepcopy(dataset.meta.stats) if dataset.meta.stats is not None else {}
    new_stats[OBS_STATE] = {key: torch.as_tensor(value) for key, value in aggregated[OBS_STATE].items()}
    new_stats[ACTION] = {key: torch.as_tensor(value) for key, value in aggregated[ACTION].items()}
    dataset.meta.stats = new_stats
    write_stats(new_stats, dataset.root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and merge LeRobot datasets.")
    parser.add_argument("merged_repo_id")
    parser.add_argument("merged_root_override")
    parser.add_argument("source_repo_ids", nargs="+")
    parser.add_argument("--normalize-height", type=int, default=360)
    parser.add_argument("--normalize-width", type=int, default=640)
    parser.add_argument("--normalize-vcodec", default="h264_nvenc")
    parser.add_argument("--expected-video-codec", default="h264")
    args = parser.parse_args()

    merged_root = Path(args.merged_root_override) if args.merged_root_override else None
    resize_shape = (args.normalize_height, args.normalize_width)

    datasets = [LeRobotDataset(repo_id) for repo_id in args.source_repo_ids]
    normalized_root_base = (
        (merged_root.parent if merged_root is not None else Path.cwd()) / ".merge_datasets_normalized"
    )
    normalized_root_base.mkdir(parents=True, exist_ok=True)

    normalized_datasets: list[LeRobotDataset] = []
    normalized_repo_ids: list[str] = []
    for dataset in datasets:
        if _dataset_matches_video_target(dataset, resize_shape, args.expected_video_codec):
            normalized_datasets.append(dataset)
            normalized_repo_ids.append(dataset.repo_id)
            continue

        normalized_repo_id = (
            f"{dataset.repo_id}_normalized_{args.normalize_width}x{args.normalize_height}_{args.expected_video_codec}"
        )
        normalized_root = normalized_root_base / _sanitize_repo_id(normalized_repo_id)
        normalized_dataset = _normalize_dataset(
            dataset=dataset,
            dst_repo_id=normalized_repo_id,
            dst_root=normalized_root,
            resize_shape=resize_shape,
            vcodec=args.normalize_vcodec,
        )
        normalized_datasets.append(normalized_dataset)
        normalized_repo_ids.append(normalized_repo_id)

    merged_dataset = merge_datasets(
        normalized_datasets,
        output_repo_id=args.merged_repo_id,
        output_dir=merged_root,
    )
    _recompute_state_action_stats(merged_dataset)
    merged_dataset.meta.info["romoya_merge"] = {
        "source_repo_ids": args.source_repo_ids,
        "normalized_repo_ids": normalized_repo_ids,
        "resize_shape": [args.normalize_height, args.normalize_width],
        "expected_video_codec": args.expected_video_codec,
        "normalize_vcodec": args.normalize_vcodec,
    }
    write_info(merged_dataset.meta.info, merged_dataset.root)


if __name__ == "__main__":
    main()

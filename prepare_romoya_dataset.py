#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import _copy_episodes_metadata_and_stats, _copy_videos, _write_parquet
from lerobot.datasets.io_utils import write_info, write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_robot_romoya.policies.configuration_act_romoya import ACTRomoyaConfig
from lerobot_robot_romoya.policies.configuration_pi05_romoya import PI05RomoyaConfig
from lerobot_robot_romoya.policies.romoya_transforms import (
    select_and_transform_action,
    select_and_transform_state,
    tensor_to_python_lists,
)


def _load_config_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return payload


def _load_policy_dict(path: Path) -> dict[str, Any]:
    payload = _load_config_payload(path)
    if "policy" in payload and isinstance(payload["policy"], dict):
        return payload["policy"]
    return payload


def _load_resize_shape(payload: dict[str, Any]) -> tuple[int, int] | None:
    dataset_cfg = payload.get("dataset")
    if not isinstance(dataset_cfg, dict):
        return None
    image_transforms = dataset_cfg.get("image_transforms")
    if not isinstance(image_transforms, dict):
        return None
    pre_tfs = image_transforms.get("pre_tfs")
    if not isinstance(pre_tfs, dict):
        return None
    resize_cfg = pre_tfs.get("resize")
    if not isinstance(resize_cfg, dict):
        return None
    kwargs = resize_cfg.get("kwargs")
    if not isinstance(kwargs, dict):
        return None
    size = kwargs.get("size")
    if not isinstance(size, list | tuple) or len(size) != 2:
        return None
    return int(size[0]), int(size[1])


def _make_policy_config(policy_type: str, policy_kwargs: dict[str, Any]):
    if policy_type == "act_romoya":
        return ACTRomoyaConfig(**policy_kwargs)
    if policy_type == "pi05_romoya":
        return PI05RomoyaConfig(**policy_kwargs)
    raise ValueError(f"Unsupported Romoya policy type for dataset prep: {policy_type}")


def _build_config(policy_dict: dict[str, Any], dataset: LeRobotDataset):
    config_input_features = dict(policy_dict.get("input_features", {}))
    for key, feature in config_input_features.items():
        if isinstance(feature, dict) and feature.get("type") == "STATE":
            config_input_features[key] = PolicyFeature(
                type=FeatureType.STATE,
                shape=tuple(feature["shape"]),
            )
        elif isinstance(feature, dict) and feature.get("type") == "VISUAL":
            config_input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=tuple(feature["shape"]),
            )

    policy_dict = dict(policy_dict)
    policy_type = policy_dict.pop("type", None)
    policy_dict.pop("type", None)
    policy_dict.pop("input_features", None)
    policy_dict.pop("output_features", None)
    policy_dict.pop("device", None)
    input_features = {
        **config_input_features,
        OBS_STATE: PolicyFeature(
            type=FeatureType.STATE,
            shape=tuple(dataset.meta.features[OBS_STATE]["shape"]),
        )
    }
    output_features = {
        ACTION: PolicyFeature(
            type=FeatureType.ACTION,
            shape=tuple(dataset.meta.features[ACTION]["shape"]),
        )
    }
    cfg = _make_policy_config(
        policy_type,
        dict(
            policy_dict,
            input_features=input_features,
            output_features=output_features,
            device="cpu",
        ),
    )
    # Dataset preparation must validate and transform against the source dataset's raw schema,
    # which can differ from the raw schema we want to preserve in saved policy configs for live inference.
    cfg.raw_observation_state_feature_names = list(dataset.meta.features[OBS_STATE].get("names", []))
    cfg.raw_action_feature_names = list(dataset.meta.features[ACTION].get("names", []))
    cfg.validate_features()
    return cfg


def _transform_file(df: pd.DataFrame, cfg: ACTRomoyaConfig) -> tuple[pd.DataFrame, dict[str, dict]]:
    raw_state = torch.tensor(df[OBS_STATE].tolist(), dtype=torch.float32)
    raw_action = torch.tensor(df[ACTION].tolist(), dtype=torch.float32)

    transformed_state = select_and_transform_state(
        raw_state,
        cfg.raw_observation_state_feature_names,
        cfg.state_feature_names,
        cfg.binary_state,
    )
    transformed_action = select_and_transform_action(
        raw_action,
        raw_state,
        cfg.raw_action_feature_names,
        cfg.raw_observation_state_feature_names,
        cfg.action_feature_names,
        cfg.binary_action,
        cfg.delta_action,
    )

    out_df = df.copy()
    out_df[OBS_STATE] = tensor_to_python_lists(transformed_state)
    out_df[ACTION] = tensor_to_python_lists(transformed_action)
    stats = compute_episode_stats(
        {
            OBS_STATE: transformed_state.numpy(),
            ACTION: transformed_action.numpy(),
        },
        {
            OBS_STATE: {"dtype": "float32", "shape": (len(cfg.state_feature_names),), "names": cfg.state_feature_names},
            ACTION: {
                "dtype": "float32",
                "shape": (len(cfg.action_feature_names),),
                "names": list(cfg.action_feature_names),
            },
        },
    )
    return out_df, stats


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
        resized = resized.round().clamp(0, 255).to(dtype=torch.uint8)
    else:
        resized = resized.clamp(0.0, 1.0).to(dtype=input_dtype)
    return resized


def _camera_shapes_match_resize_target(dataset: LeRobotDataset, resize_shape: tuple[int, int]) -> bool:
    resize_h, resize_w = resize_shape
    for key in dataset.meta.camera_keys:
        feature = dataset.meta.features.get(key, {})
        shape = tuple(feature.get("shape", ()))
        if len(shape) != 3:
            return False
        height, width = shape[0], shape[1]
        if (height, width) != (resize_h, resize_w):
            return False
    return True


def _transform_and_resize_dataset(
    dataset: LeRobotDataset,
    cfg: ACTRomoyaConfig,
    new_features: dict[str, dict[str, Any]],
    resize_shape: tuple[int, int],
    dst_repo_id: str,
    dst_root: str | None,
) -> LeRobotDataset:
    use_videos = bool(dataset.meta.video_keys)
    out_dataset = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.info.get("robot_type"),
        root=dst_root,
        use_videos=use_videos,
        tolerance_s=dataset.tolerance_s,
        vcodec=dataset.vcodec,
    )

    camera_keys = list(dataset.meta.camera_keys)
    feature_keys = [key for key in dataset.features if key not in DEFAULT_FEATURES]

    dataset._ensure_hf_dataset_loaded()
    num_episodes = len(dataset.meta.episodes["dataset_from_index"])
    for episode_idx in range(num_episodes):
        start = int(dataset.meta.episodes["dataset_from_index"][episode_idx])
        end = int(dataset.meta.episodes["dataset_to_index"][episode_idx])
        for frame_idx in range(start, end):
            item = dataset[frame_idx]

            raw_state = item[OBS_STATE].to(dtype=torch.float32)
            raw_action = item[ACTION].to(dtype=torch.float32)
            transformed_state = select_and_transform_state(
                raw_state,
                cfg.raw_observation_state_feature_names,
                cfg.state_feature_names,
                cfg.binary_state,
            )
            transformed_action = select_and_transform_action(
                raw_action,
                raw_state,
                cfg.raw_action_feature_names,
                cfg.raw_observation_state_feature_names,
                cfg.action_feature_names,
                cfg.binary_action,
                cfg.delta_action,
            )

            frame = {
                "task": item["task"],
            }
            for key in feature_keys:
                if key in {OBS_STATE, ACTION, "task_index", "frame_index", "episode_index", "index"}:
                    continue
                if key in camera_keys:
                    resized_image = _resize_image_tensor(item[key], resize_shape)
                    frame[key] = resized_image.permute(1, 2, 0).contiguous().cpu().numpy()
                else:
                    frame[key] = item[key]
            frame[OBS_STATE] = transformed_state
            frame[ACTION] = transformed_action
            out_dataset.add_frame(frame)

        out_dataset.save_episode()

    out_dataset.finalize()
    return out_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a transformed Romoya training dataset.")
    parser.add_argument("--src-repo-id", required=True)
    parser.add_argument("--dst-repo-id", required=True)
    parser.add_argument("--config-path", required=True, help="Train config JSON or policy config JSON.")
    parser.add_argument("--src-root")
    parser.add_argument("--dst-root")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--resize-height", type=int)
    parser.add_argument("--resize-width", type=int)
    args = parser.parse_args()

    dataset = LeRobotDataset(
        args.src_repo_id,
        root=args.src_root,
        download_videos=not args.skip_videos,
    )
    config_payload = _load_config_payload(Path(args.config_path))
    policy_dict = config_payload["policy"] if "policy" in config_payload else config_payload
    if policy_dict.get("type") not in {"act_romoya", "pi05_romoya"}:
        raise ValueError("prepare_romoya_dataset.py only supports policy.type=act_romoya or policy.type=pi05_romoya.")
    cfg = _build_config(policy_dict, dataset)

    resize_shape = None
    if args.resize_height is not None or args.resize_width is not None:
        if args.resize_height is None or args.resize_width is None:
            raise ValueError("Both --resize-height and --resize-width must be provided together.")
        resize_shape = (args.resize_height, args.resize_width)
    else:
        resize_shape = _load_resize_shape(config_payload)
    should_resize_visuals = resize_shape is not None and not _camera_shapes_match_resize_target(dataset, resize_shape)

    if should_resize_visuals and args.skip_videos:
        raise ValueError("Resizing during Romoya dataset preparation is not supported together with --skip-videos.")

    new_features = copy.deepcopy(dataset.meta.features)
    new_features[OBS_STATE] = {
        **new_features[OBS_STATE],
        "shape": (len(cfg.state_feature_names),),
        "names": list(cfg.state_feature_names),
    }
    new_features[ACTION] = {
        **new_features[ACTION],
        "shape": (len(cfg.action_feature_names),),
        "names": list(cfg.action_feature_names),
    }
    if should_resize_visuals:
        resize_h, resize_w = resize_shape
        for key in dataset.meta.camera_keys:
            feature = dict(new_features[key])
            channels = feature["shape"][-1]
            feature["shape"] = (resize_h, resize_w, channels)
            new_features[key] = feature

    if should_resize_visuals:
        transformed_dataset = _transform_and_resize_dataset(
            dataset=dataset,
            cfg=cfg,
            new_features=new_features,
            resize_shape=resize_shape,
            dst_repo_id=args.dst_repo_id,
            dst_root=args.dst_root,
        )
        new_meta = transformed_dataset.meta
    else:
        new_meta = LeRobotDatasetMetadata.create(
            repo_id=args.dst_repo_id,
            fps=dataset.meta.fps,
            features=new_features,
            robot_type=dataset.meta.info.get("robot_type"),
            root=args.dst_root,
            use_videos=bool(dataset.meta.video_keys) and not args.skip_videos,
            chunks_size=dataset.meta.chunks_size,
            data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
            video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
        )

        stats_list: list[dict[str, dict]] = []
        parquet_files = sorted((dataset.root / "data").glob("*/*.parquet"))
        for src_path in parquet_files:
            df = pd.read_parquet(src_path).reset_index(drop=True)
            transformed_df, file_stats = _transform_file(df, cfg)
            dst_path = new_meta.root / src_path.relative_to(dataset.root)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            _write_parquet(transformed_df, dst_path, new_meta)
            stats_list.append(file_stats)

        if new_meta.video_keys and not args.skip_videos:
            _copy_videos(dataset, new_meta)
        _copy_episodes_metadata_and_stats(dataset, new_meta)

        new_stats = copy.deepcopy(dataset.meta.stats)
        aggregated = aggregate_stats(stats_list)
        new_stats[OBS_STATE] = {key: torch.as_tensor(value) for key, value in aggregated[OBS_STATE].items()}
        new_stats[ACTION] = {key: torch.as_tensor(value) for key, value in aggregated[ACTION].items()}
        new_meta.stats = new_stats
        write_stats(new_stats, new_meta.root)

    new_meta.info["romoya_prepare"] = {
        "action_mode": getattr(cfg, "action_mode", None),
        "state_feature_names": list(cfg.state_feature_names),
        "action_feature_names": list(cfg.action_feature_names),
        "binary_state": list(cfg.binary_state),
        "binary_action": list(cfg.binary_action),
        "delta_action": list(cfg.delta_action),
        "control_schema": cfg.control_schema,
        "source_repo_id": args.src_repo_id,
        "source_raw_observation_state_feature_names": list(cfg.raw_observation_state_feature_names),
        "source_raw_action_feature_names": list(cfg.raw_action_feature_names),
        "resize_shape": list(resize_shape) if resize_shape is not None else None,
    }
    write_info(new_meta.info, new_meta.root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a saved policy on a local LeRobot dataset and report action error.

This uses the inference path:
  observation -> preprocessor -> policy.select_action -> postprocessor

and compares the reconstructed robot action against the recorded dataset action.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins

TCP_ACTION_MODES = {"absolute_tcp_gripper_do1", "delta_tcp_gripper_do1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo-id", type=str, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    return parser.parse_args()


def to_float_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def should_keep_frame(item: dict, allowed_episodes: set[int] | None) -> bool:
    if allowed_episodes is None:
        return True
    return int(item["episode_index"]) in allowed_episodes


def main() -> None:
    args = parse_args()
    print(f"Loading dataset: {args.dataset_repo_id}")
    if args.dataset_root is not None:
        print(f"Using dataset root override: {args.dataset_root}")
    print(f"Loading policy: {args.policy_path}")
    print(f"Using device: {args.device}")
    register_third_party_plugins()
    try:
        import lerobot_robot_romoya  # noqa: F401
        from lerobot_robot_romoya.policies.configuration_act_romoya import ACTRomoyaConfig  # noqa: F401
    except ImportError:
        pass

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    print(f"Dataset loaded: {len(dataset)} frames, {dataset.num_episodes} episodes")
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = args.device

    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
    )
    print("Policy and processors loaded.")

    device = torch.device(policy_cfg.device)
    policy.reset()
    print(f"Policy running on device: {device}")

    allowed_episodes = set(args.episodes) if args.episodes else None
    last_episode_index: int | None = None
    compare_in_transformed_policy_space = getattr(policy_cfg, "action_mode", None) in TCP_ACTION_MODES
    frames_evaluated = 0
    total_abs_error = None
    total_sq_error = None
    total_count = 0

    total_frames = args.max_frames if args.max_frames is not None else len(dataset)
    progress = tqdm(total=total_frames, desc="Evaluating", unit="frame")

    for idx in range(len(dataset)):
        item = dataset[idx]
        if not should_keep_frame(item, allowed_episodes):
            continue

        episode_index = int(item["episode_index"])
        if last_episode_index != episode_index:
            policy.reset()
            last_episode_index = episode_index
            progress.set_postfix(episode=episode_index)

        observation = {key: value for key, value in item.items() if key != ACTION}
        if compare_in_transformed_policy_space:
            batch = {**observation, ACTION: item[ACTION]}
            processed = preprocessor(batch)
            model_observation = {key: value for key, value in processed.items() if key != ACTION}
            with torch.inference_mode():
                predicted_action = policy.select_action(model_observation).to("cpu").squeeze(0)
            target_action = to_float_tensor(processed[ACTION]).to("cpu")
            target_action = target_action.squeeze(0) if target_action.ndim > 1 else target_action
        else:
            observation = preprocessor(observation)
            with torch.inference_mode():
                predicted_action = policy.select_action(observation)
            predicted_action = postprocessor(predicted_action).to("cpu").squeeze(0)
            target_action = to_float_tensor(item[ACTION]).to("cpu")
            target_action = target_action.squeeze(0) if target_action.ndim > 1 else target_action

        abs_error = (predicted_action - target_action).abs()
        sq_error = (predicted_action - target_action).pow(2)

        if total_abs_error is None:
            total_abs_error = torch.zeros_like(abs_error)
            total_sq_error = torch.zeros_like(sq_error)

        total_abs_error += abs_error
        total_sq_error += sq_error
        total_count += 1
        frames_evaluated += 1
        progress.update(1)

        if args.max_frames is not None and frames_evaluated >= args.max_frames:
            break

    progress.close()

    if total_count == 0 or total_abs_error is None or total_sq_error is None:
        raise SystemExit("No frames evaluated.")

    mae = total_abs_error / total_count
    rmse = torch.sqrt(total_sq_error / total_count)

    action_names = (
        list(getattr(policy_cfg, "transformed_action_names"))
        if compare_in_transformed_policy_space
        else dataset.meta.features[ACTION]["names"]
    )
    print(f"Frames evaluated: {total_count}")
    if compare_in_transformed_policy_space:
        print("Comparing in transformed policy action space.")
    print("Per-dimension MAE:")
    for name, value in zip(action_names, mae.tolist(), strict=True):
        print(f"  {name}: {value:.6f}")
    print("Per-dimension RMSE:")
    for name, value in zip(action_names, rmse.tolist(), strict=True):
        print(f"  {name}: {value:.6f}")
    print(f"Mean MAE over all action dims: {mae.mean().item():.6f}")
    print(f"Mean RMSE over all action dims: {rmse.mean().item():.6f}")


if __name__ == "__main__":
    main()

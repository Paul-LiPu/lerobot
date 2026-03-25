#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from collections.abc import Iterator

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                ep_length = end_index - start_index
                if drop_n_first_frames + drop_n_last_frames >= ep_length:
                    logger.warning(
                        "Episode %d has %d frames but drop_n_first_frames=%d and "
                        "drop_n_last_frames=%d removes all frames. Skipping.",
                        episode_idx,
                        ep_length,
                        drop_n_first_frames,
                        drop_n_last_frames,
                    )
                    continue
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        if not indices:
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class RangeEventSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        observation_states: Tensor,
        state_feature_names: list[str],
        event_state_names: list[str],
        event_low: float,
        event_high: float,
        event_horizon: int,
        event_probability: float,
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")
        if event_horizon < 0:
            raise ValueError(f"event_horizon must be >= 0, got {event_horizon}")
        if not 0.0 <= event_probability <= 1.0:
            raise ValueError(f"event_probability must be in [0, 1], got {event_probability}")
        if event_low >= event_high:
            raise ValueError(f"event_low must be < event_high, got {event_low} >= {event_high}")
        if len(event_state_names) == 0:
            raise ValueError("event_state_names must be non-empty.")

        missing_state_names = [name for name in event_state_names if name not in state_feature_names]
        if missing_state_names:
            raise ValueError(
                "event_state_names are missing from transformed observation.state names. "
                f"Missing: {missing_state_names}. Available: {state_feature_names}"
            )

        if observation_states.ndim != 2:
            raise ValueError(
                f"observation_states is expected to be rank-2 [num_frames, state_dim], got {tuple(observation_states.shape)}"
            )
        if observation_states.shape[1] != len(state_feature_names):
            raise ValueError(
                "observation_states.shape[1] must match len(state_feature_names), got "
                f"{observation_states.shape[1]} != {len(state_feature_names)}"
            )

        self.event_probability = event_probability
        self.shuffle = shuffle
        self.event_state_names = list(event_state_names)
        self.event_low = event_low
        self.event_high = event_high
        self.event_horizon = event_horizon

        state_index = {name: idx for idx, name in enumerate(state_feature_names)}
        event_state_indices = [state_index[name] for name in event_state_names]
        event_states = observation_states[:, event_state_indices]

        indices = []
        special_indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is not None and episode_idx not in episode_indices_to_use:
                continue

            valid_start = start_index + drop_n_first_frames
            valid_end = end_index - drop_n_last_frames
            ep_length = end_index - start_index
            if valid_start >= valid_end:
                logger.warning(
                    "Episode %d has %d frames but drop_n_first_frames=%d and drop_n_last_frames=%d removes all frames. Skipping.",
                    episode_idx,
                    ep_length,
                    drop_n_first_frames,
                    drop_n_last_frames,
                )
                continue

            for frame_idx in range(valid_start, valid_end):
                indices.append(frame_idx)
                horizon_end = min(end_index, frame_idx + event_horizon + 1)
                window = event_states[frame_idx:horizon_end]
                if torch.any((window > event_low) & (window < event_high)):
                    special_indices.append(frame_idx)

        if not indices:
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self.indices = indices
        self.special_indices = special_indices
        self.special_fraction = len(self.special_indices) / len(self.indices)

        if len(self.special_indices) == 0:
            logger.warning(
                "RangeEventSampler found no special-event frames for state_names=%s, range=(%s, %s), horizon=%d. Falling back to uniform sampling.",
                self.event_state_names,
                self.event_low,
                self.event_high,
                self.event_horizon,
            )

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            for i in self.indices:
                yield i
            return

        all_indices = torch.tensor(self.indices, dtype=torch.int64)
        special_indices = (
            torch.tensor(self.special_indices, dtype=torch.int64) if self.special_indices else None
        )
        for _ in range(len(self.indices)):
            if special_indices is not None and torch.rand(()) < self.event_probability:
                sample_pool = special_indices
            else:
                sample_pool = all_indices
            sampled_idx = int(torch.randint(len(sample_pool), size=(1,)).item())
            yield int(sample_pool[sampled_idx].item())

    def __len__(self) -> int:
        return len(self.indices)

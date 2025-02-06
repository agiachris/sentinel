from typing import Any, Dict, Union, List, Generator, Optional

import os
import glob
import pickle
import pathlib
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset
from copy import deepcopy


def load_pickle(path: Union[str, pathlib.Path]) -> pd.DataFrame:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


class EpisodeDataset(IterableDataset):

    def __init__(
        self,
        dataset_path: Union[str, pathlib.Path],
        exec_horizon: int,
        sample_history: int,
        filter_success: bool = False,
        filter_failure: bool = False,
        filter_episodes: Optional[List[int]] = None,
        max_episode_length: Optional[int] = None,
        max_num_episodes: Optional[int] = None,
    ) -> None:
        """Construct EpisodeDataset."""
        super().__init__()
        assert exec_horizon >= 1 and sample_history >= 0
        self._dataset_path = dataset_path
        self._episode_files = sorted(glob.glob(os.path.join(dataset_path, "*")))
        self._exec_horizon = exec_horizon
        self._sample_history = sample_history
        self._filter_success = filter_success
        self._filter_failure = filter_failure
        self._filter_episodes = filter_episodes
        self._max_episode_length = max_episode_length
        self._max_num_episodes = max_num_episodes

    def __iter__(
        self,
    ) -> Generator[Union[Dict[str, Any], List[Dict[str, Any]]], None, None]:
        """Return sample."""
        num_episodes = 0
        for i, file_path in enumerate(self._episode_files):
            # if self._max_num_episodes is not None and num_episodes >= self._max_num_episodes:
            if self._max_num_episodes is not None and i >= self._max_num_episodes:
                continue

            episode = load_pickle(file_path)
            success = episode.iloc[0].to_dict().get("success", True)
            if (
                (self._filter_success and success)
                or (self._filter_failure and not success)
                or (
                    self._filter_episodes is not None
                    and not isinstance(self._filter_episodes, str)
                    and i in self._filter_episodes
                )
            ):
                continue
            else:
                num_episodes += 1

            for idx in range(
                self._exec_horizon * self._sample_history,
                len(episode),
                self._exec_horizon,
            ):
                if (
                    self._max_episode_length is not None
                    and episode.iloc[idx].to_dict()["timestep"]
                    >= self._max_episode_length
                ):
                    continue

                sample = [
                    episode.iloc[j].to_dict()
                    for j in range(
                        idx - self._exec_horizon * self._sample_history,
                        idx + 1,
                        self._exec_horizon,
                    )
                ]
                assert all(x["episode"] == i for x in sample)
                yield sample[0] if len(sample) == 1 else sample


class VideoDataset(IterableDataset):

    def __init__(
        self,
        dataset_path: Union[str, pathlib.Path],
        exec_horizon: int,
        num_timesteps: int,
        subsample_freq: int = 1,
        filter_success: bool = False,
        filter_failure: bool = False,
        filter_episodes: Optional[List[int]] = None,
        max_episode_length: Optional[int] = None,
        max_num_episodes: Optional[int] = None,
    ) -> None:
        """Construct VideoDataset."""
        super().__init__()
        assert exec_horizon >= 1
        self._dataset_path = dataset_path
        self._episode_files = sorted(glob.glob(os.path.join(dataset_path, "*")))
        self._exec_horizon = exec_horizon
        self._num_timesteps = num_timesteps
        self._subsample_freq = subsample_freq
        self._filter_success = filter_success
        self._filter_failure = filter_failure
        self._filter_episodes = filter_episodes
        self._max_episode_length = max_episode_length
        self._max_num_episodes = max_num_episodes

    def get_episode(
        self,
        episode: int,
        crop_episode_length: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return list of dictionaries for each timestep in the specified episode."""
        episode_frame: pd.DataFrame = load_pickle(self._episode_files[episode])
        assert episode_frame.iloc[0].to_dict()["episode"] == episode
        episode_data = [
            episode_frame.iloc[i].to_dict() for i in range(len(episode_frame))
        ]
        if crop_episode_length:
            episode_data = [
                x for x in episode_data if x["timestep"] < self._max_episode_length
            ]
        # Keep last frame for goal state reasoning.
        episode_data = episode_data[::-1][:: self._exec_horizon * self._subsample_freq][
            ::-1
        ]
        return episode_data

    def __iter__(
        self,
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Return sample."""
        for i in range(len(self._episode_files)):
            if self._max_num_episodes is not None and i >= self._max_num_episodes:
                continue

            episode_data = self.get_episode(i, crop_episode_length=True)
            success = episode_data[0].get("success", True)
            if (
                (self._filter_success and success)
                or (self._filter_failure and not success)
                or (
                    self._filter_episodes is not None
                    and not isinstance(self._filter_episodes, str)
                    and i in self._filter_episodes
                )
            ):
                continue

            episode_length = len(episode_data)
            episode_ratios = np.linspace(0, 1, self._num_timesteps + 1)[1:]
            for ratio in episode_ratios:
                idx = int(ratio * episode_length)
                sample = deepcopy(episode_data[:idx])
                assert all(x["episode"] == i for x in sample)
                yield sample

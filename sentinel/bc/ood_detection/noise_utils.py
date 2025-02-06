from typing import Union

import abc
import numpy as np


class ActionNoise(abc.ABC):

    def __init__(
        self,
        num_robots: int = 2,
        action_dim: int = 4,
    ):
        """Construct ActionNoise."""
        self._num_robots = num_robots
        self._action_dim = action_dim

    def apply_noise(self, action: np.ndarray) -> np.ndarray:
        """Apply noise to action."""
        if action.shape != (self._num_robots, self._action_dim):
            raise ValueError(f"Unexpected action with shape {action.shape}.")
        _action = action.copy()

        start_idx = 1 if self._action_dim > 3 else 0
        _action[:, start_idx : start_idx + 3] += self._sample_noise(
            _action[:, start_idx : start_idx + 3]
        )

        return _action

    @abc.abstractmethod
    def _sample_noise(self, action: np.ndarray) -> Union[float, np.ndarray]:
        """Sample noise to apply to action."""
        raise NotImplementedError


class UniformNoise(ActionNoise):
    def __init__(self, lower_bound: float = -0.1, upper_bound: float = 0.1, **kwargs):
        """Construct UniformNoise."""
        super().__init__(**kwargs)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def _sample_noise(self, action: np.ndarray) -> np.ndarray:
        return np.random.uniform(self.lower_bound, self.upper_bound, action.shape)


class GaussianNoise(ActionNoise):
    def __init__(self, mean: float = 0.0, std_dev: float = 0.1, **kwargs):
        """Construct GaussianNoise."""
        super().__init__(**kwargs)
        self.mean = mean
        self.std_dev = std_dev

    def _sample_noise(self, action: np.ndarray) -> np.ndarray:
        return np.random.normal(self.mean, self.std_dev, action.shape)

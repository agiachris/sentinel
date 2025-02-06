from typing import Callable, Type

import numpy as np

from .detectors import (
    guassian_detector,
    quantile_detector,
)
from .noise_utils import (
    ActionNoise,
    UniformNoise,
    GaussianNoise,
)


def get_detector(detector: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return reconstruction detector."""
    if detector == "gaussian":
        return guassian_detector
    elif detector == "quantile":
        return quantile_detector

    raise ValueError(f"Detector {detector} is not supported.")


def get_noise(noise: str) -> Type[ActionNoise]:
    """Return action noise scheduler."""
    if noise == "UniformNoiseScheduler":
        return UniformNoise
    elif noise == "GaussianNoiseScheduler":
        return GaussianNoise

    raise ValueError(f"Noise {noise} is not supported.")

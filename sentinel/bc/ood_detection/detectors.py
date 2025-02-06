import numpy as np
from scipy import stats


def guassian_detector(
    calib_scores: np.ndarray,
    test_scores: np.ndarray,
    quantile: float = 0.95,
) -> np.ndarray:
    """Return predictions and scores for Gaussian detector."""
    z_score = stats.norm.ppf(quantile)
    threshold = calib_scores.mean() + z_score * calib_scores.std()
    return test_scores >= threshold


def quantile_detector(
    calib_scores: np.ndarray,
    test_scores: np.ndarray,
    quantile: float = 0.95,
) -> np.ndarray:
    """Return predictions and scores for quantile detector."""
    threshold = np.quantile(calib_scores, quantile)
    return test_scores >= threshold

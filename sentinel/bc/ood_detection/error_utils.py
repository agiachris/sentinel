from typing import List, Dict, Callable, Optional, Union, Any

import numpy as np
from sklearn import metrics
from sklearn.neighbors import KernelDensity

from .action_utils import (
    filter_actions,
    filter_gripper_action,
    filter_rotation_actions,
    velocity_to_trajectory,
)


def compute_ate(x: List[np.ndarray], y: List[np.ndarray]) -> Union[float, np.ndarray]:
    """Compute average trajectory error (ATE)."""
    assert len(x) == len(y), "Both input arrays must have the same shape."
    error = 0
    for _x, _y in zip(x, y):
        distances = np.linalg.norm(_x - _y, axis=-1)
        error += np.mean(distances, axis=-1)
    return error


def compute_l2_distance(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Compute L2-norm between all pairs of vectors.

    Args:
        x: (N, D) matrix.
        y: (M, D) matrix.

    Returns:
        (N, M) matrix.
    """
    assert x.ndim == 2 and y.ndim == 2
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)


def compute_cosine_error(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between all pairs of vectors.

    Args:
        x: (N, D) matrix.
        y: (M, D) matrix.

    Returns:
        (N, M) matrix.
    """
    assert x.ndim == 2 and y.ndim == 2
    x_norm: np.ndarray = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm: np.ndarray = np.linalg.norm(y, axis=1, keepdims=True)
    assert np.all(x_norm > 0) and np.all(y_norm > 0)
    return -1.0 * (x / x_norm) @ (y / y_norm).T


def compute_mmd_rbf(
    x: np.ndarray, y: np.ndarray, gamma: Optional[Union[float, str]] = None
) -> np.float64:
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Args:
        x: [N, D] matrix.
        y: [M, D] matrix.
        gamma: rbf kernel parameter.

    Returns:
        MMD value.
    """
    assert x.ndim == 2 and y.ndim == 2

    if isinstance(gamma, str):
        if gamma == "median":
            z = np.vstack([x, y])
            distances = np.sum((z[:, np.newaxis, :] - z[np.newaxis, :, :]) ** 2, axis=2)
            gamma = 1.0 / (2 * np.median(distances[distances > 0]))
        elif gamma == "max_eig":
            z = np.vstack([x, y])
            cov = np.cov(z.T)
            max_eig = np.max(np.linalg.eigvalsh(cov))
            gamma = 1.0 / max_eig
        else:
            raise ValueError(f"Gamma {gamma} is not supported.")

    xx = metrics.pairwise.rbf_kernel(x, x, gamma)
    yy = metrics.pairwise.rbf_kernel(y, y, gamma)
    xy = metrics.pairwise.rbf_kernel(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_kde_kl(
    x: np.ndarray,
    y: np.ndarray,
    kernel: str = "gaussian",
    bandwidth: Union[float, str] = 1.0,
    forward: bool = True,
) -> np.float64:
    """KL divergence using KDE approximation of marginal distribution.

    Args:
        x: [N, D] matrix.
        y: [M, D] matrix.
        bandwidth: KDE parameter.

    Returns:
        KL divergence.
    """
    assert x.ndim == 2 and y.ndim == 2

    if isinstance(bandwidth, str):
        if bandwidth == "max_eig":
            z = np.vstack([x, y])
            cov = np.cov(z.T)
            max_eig = np.max(np.linalg.eigvalsh(cov))
            bandwidth = np.sqrt(max_eig)
        else:
            raise ValueError(f"Bandwidth {bandwidth} is not supported.")

    p: KernelDensity = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(x)
    q: KernelDensity = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(y)
    if forward:
        # If defined in terms of q(y): Reverse KL.
        # Low if current is mode seeking.
        # High if current is mode covering.
        log_p = p.score_samples(x)
        log_q = q.score_samples(x)
        kl_est = np.mean(log_p - log_q)
    else:
        # If defined in terms of q(y): Forward KL.
        # Low if current is mode covering.
        # High if current is mode seeking.
        log_q = q.score_samples(y)
        log_p = p.score_samples(y)
        kl_est = np.mean(log_q - log_p)
    return kl_est


CONSISTENCY_ERROR_FNS: Dict[
    str, Callable[[np.ndarray, np.ndarray], Union[float, np.ndarray]]
] = {
    "ssd": lambda x, y: ((x - y) ** 2).sum(axis=(-1, -2)),
    "mse": lambda x, y: ((x - y) ** 2).mean(axis=(-1, -2)),
    "ate": compute_ate,
}


CONSISTENCY_DIST_ERROR_FNS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mmd_rbf": compute_mmd_rbf,
    "kde_kl": compute_kde_kl,
}


TRAJECTORY_ERRORS = {"ate"}


EMBEDDING_ERROR_FNS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "l2": compute_l2_distance,
    "cosine": compute_cosine_error,
}


def compute_temporal_error(
    error_fn: str,
    curr_action: np.ndarray,
    prev_action: np.ndarray,
    pred_horizon: int,
    exec_horizon: int,
    ignore_gripper: bool = True,
    ignore_rotation: bool = True,
    sim_freq: float = 5,
    num_robots: int = 2,
    action_dim: int = 4,
    pos_actions: bool = False,
    skip_steps: bool = False,
    curr_skip_steps: Optional[int] = None,
    prev_skip_steps: Optional[int] = None,
    curr_action_gt: Optional[np.ndarray] = None,
    prev_action_gt: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Union[float, np.ndarray]:
    """Compute temporal consistency error.

    Args:
        error_fn: Action error function.
        curr_action: Current action (Optional: batch_size, pred_horizon, action_dim).
        prev_action: Previous action (Optional: batch_size, pred_horizon, action_dim).
        pred_horizon: Action prediction horizon.
        exec_horizon: Robot execution horizon.
        ignore_gripper: Filter robot gripper command.
        ignore_rotation: Filter rotation commands.
        sim_freq: Simulation environment frequency.
        num_robots: Number of robot end-effectors.
        action_dim: Action dimensionality.
        pos_actions: Whether actions are positions.
        skip_steps: Whether to skip first few steps of action sequences.
        curr_skip_steps: Skip the first few steps of curr action sequence.
        prev_skip_steps: Skip the first few steps of prev action sequence.
        curr_action_gt: Ground truth current action.
        prev_action_gt: Ground truth previous action.
        kwargs: Keyword arguments for error function.
    """
    assert pred_horizon - exec_horizon > 0, "No overlap in action prediction horizon."
    if curr_action_gt is not None and prev_action_gt is not None:
        assert np.all(curr_action_gt[:-exec_horizon] == prev_action_gt[exec_horizon:])

    is_batch = False
    if error_fn in CONSISTENCY_ERROR_FNS:
        if prev_action.ndim == 2:
            prev_action = prev_action[None, ...]
        if curr_action.ndim == 2:
            curr_action = curr_action[None, ...]
        elif curr_action.ndim == 3:
            is_batch = True

    assert curr_action.ndim == 3 and prev_action.ndim == 3
    assert curr_action.shape[1:] == prev_action.shape[1:]
    assert pred_horizon <= prev_action.shape[1]

    # Extract overlapping actions.
    if skip_steps:
        assert curr_skip_steps is not None and prev_skip_steps is not None
        prev = prev_action[:, prev_skip_steps + exec_horizon : pred_horizon]
        curr = curr_action[
            :,
            curr_skip_steps : pred_horizon
            - exec_horizon
            + curr_skip_steps
            - prev_skip_steps,
        ]
    else:
        prev = prev_action[:, exec_horizon:pred_horizon]
        curr = curr_action[:, : pred_horizon - exec_horizon]

    if pos_actions:
        raise ValueError("Pos actions is not currently supported.")

    if action_dim >= 4:
        # Filter gripper actions.
        filtered_gripper = False
        if (not pos_actions) and (ignore_gripper or error_fn in TRAJECTORY_ERRORS):
            curr = filter_gripper_action(
                curr, num_robots=num_robots, action_dim=action_dim
            )
            prev = filter_gripper_action(
                prev, num_robots=num_robots, action_dim=action_dim
            )
            filtered_gripper = True

        # Filter rotation actions.
        if (not pos_actions) and (ignore_rotation or error_fn in TRAJECTORY_ERRORS):
            if filtered_gripper:
                action_dim = action_dim - 1
            curr = filter_rotation_actions(
                curr, num_robots=num_robots, action_dim=action_dim
            )
            prev = filter_rotation_actions(
                prev, num_robots=num_robots, action_dim=action_dim
            )

        # Convert velocities to trajectories.
        if error_fn in TRAJECTORY_ERRORS:
            assert ignore_gripper and ignore_rotation
            curr = velocity_to_trajectory(
                curr, time=1 / sim_freq, num_robots=num_robots
            )
            prev = velocity_to_trajectory(
                prev, time=1 / sim_freq, num_robots=num_robots
            )

    if error_fn in CONSISTENCY_ERROR_FNS:
        error_fn = CONSISTENCY_ERROR_FNS[error_fn]
        error: np.ndarray = error_fn(curr, prev)
    elif error_fn in CONSISTENCY_DIST_ERROR_FNS:
        error_fn = CONSISTENCY_DIST_ERROR_FNS[error_fn]
        curr = curr.reshape(curr.shape[0], -1)
        prev = prev.reshape(prev.shape[0], -1)
        error: np.ndarray = error_fn(curr, prev, **kwargs)
    else:
        raise ValueError(f"Error function {error_fn} does not exist.")

    return error if is_batch else error.item()


def compute_action_variance(
    actions: np.ndarray,
    pred_horizon: int,
    ignore_gripper: bool = True,
    ignore_rotation: bool = True,
    use_trajectory: bool = False,
    sim_freq: float = 5,
    num_robots: int = 2,
    action_dim: int = 4,
) -> float:
    """Compute variance across a batch of actions.

    Args:
        actions: Batch of actions (batch_size, pred_horizon, action_dim).
        pred_horizon: Action prediction horizon.
        ignore_gripper: Filter robot gripper command.
        ignore_rotation: Filter rotation commands.
        sim_freq: Simulation environment frequency.
        num_robots: Number of robot end-effectors.
        action_dim: Action dimensionality.
    """
    assert actions.ndim == 3, "Must be a batch of actions."
    assert pred_horizon <= actions.shape[1]

    # Extract prediction horizon.
    actions = actions[:, :pred_horizon]

    if action_dim >= 4:
        # Filter gripper and rotation actions.
        actions = filter_actions(
            actions,
            num_robots=num_robots,
            action_dim=action_dim,
            ignore_gripper=ignore_gripper,
            ignore_rotation=ignore_rotation,
        )

        # Convert velocities to trajectories.
        if use_trajectory:
            assert ignore_gripper and ignore_rotation
            actions = velocity_to_trajectory(
                actions, time=1 / sim_freq, num_robots=num_robots, return_list=False
            )

    variance: np.ndarray = actions.var(axis=0).mean()
    return variance.item()


def topk_embeddings(
    data_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    error_fn: str,
    k: int,
    leave_one_out: bool = False,
) -> np.ndarray:
    """Compute top-k embedding similarity scores."""
    scores = EMBEDDING_ERROR_FNS[error_fn](
        data_embeddings,
        test_embeddings,
    )

    if leave_one_out:
        assert data_embeddings.shape == test_embeddings.shape
        scores += np.diag([np.inf] * data_embeddings.shape[0])

    scores = np.sort(scores, axis=0)
    scores = np.mean(scores[:k], axis=0)

    return scores


def mahal_embeddings(
    data_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute Mahalanobis embedding similarity scores."""
    mean = np.mean(data_embeddings, axis=0)
    cov = np.cov(data_embeddings.T)
    cov += np.eye(cov.shape[0]) * 1e-12
    invcov = np.linalg.inv(cov)

    mahals = (test_embeddings - mean) @ invcov.T
    mahals = np.sum((test_embeddings - mean) * mahals, axis=1)
    mahals = np.sqrt(mahals)

    return mahals


def compute_embedding_scores(
    data_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    method: str,
    method_kwargs: Optional[Dict[str, Any]] = None,
    leave_one_out: bool = False,
) -> np.ndarray:
    """Compute embedding scores for test data."""
    if data_embeddings.ndim == 1 and isinstance(data_embeddings[0], np.ndarray):
        data_embeddings = np.stack(data_embeddings)
    if test_embeddings.ndim == 1 and isinstance(test_embeddings[0], np.ndarray):
        test_embeddings = np.stack(test_embeddings)

    if method == "topk":
        scores = topk_embeddings(
            data_embeddings=data_embeddings,
            test_embeddings=test_embeddings,
            leave_one_out=leave_one_out,
            **method_kwargs,
        )

    elif method == "mahal":
        scores = mahal_embeddings(
            data_embeddings=data_embeddings,
            test_embeddings=test_embeddings,
        )
    else:
        raise ValueError(f"Embedding score method {method} is not supported.")

    return scores

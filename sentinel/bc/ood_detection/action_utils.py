from typing import List, Union, List

import numpy as np

from sentinel.bc import utils


def velocity_to_trajectory(
    action: np.ndarray,
    time: float = 1 / 3,
    num_robots: int = 2,
    return_list: bool = True,
) -> List[np.ndarray]:
    """Compute trajectory from velocity vector."""
    assert action.shape[-1] == 3 * num_robots, "Must be delta positions."
    trajectory = (action * time).cumsum(axis=-2)
    if not return_list:
        return trajectory
    return [trajectory[..., i * 3 : (i + 1) * 3] for i in range(num_robots)]


def filter_gripper_action(
    action: np.ndarray,
    num_robots: int = 2,
    action_dim: int = 4,
) -> np.ndarray:
    """Return actions without (binary) gripper command."""
    if action_dim >= 4:
        mask = np.ones(action_dim * num_robots, dtype=bool)
        assert mask.shape[0] == action.shape[-1]
        mask[::action_dim] = False
        action = action[..., mask]
        assert action.shape[-1] == num_robots * (action_dim - 1)
    return action


def filter_rotation_actions(
    action: np.ndarray,
    num_robots: int = 2,
    action_dim: int = 4,
) -> np.ndarray:
    """Return actions without rotation commands."""
    if action_dim >= 4:
        mask = np.ones(action_dim * num_robots, dtype=bool)
        assert mask.shape[0] == action.shape[-1]
        mask[action_dim - 1 :: action_dim] = False
        mask[action_dim - 2 :: action_dim] = False
        mask[action_dim - 3 :: action_dim] = False
        action = action[..., mask]
        assert action.shape[-1] == num_robots * (action_dim - 3)
    return action


def filter_actions(
    action: np.ndarray,
    num_robots: int = 2,
    action_dim: int = 4,
    ignore_gripper: bool = True,
    ignore_rotation: bool = True,
) -> np.ndarray:
    """Return actions without (binary) gripper and rotation commands."""
    if action_dim >= 4:

        if ignore_gripper:
            action = filter_gripper_action(
                action, num_robots=num_robots, action_dim=action_dim
            )
            action_dim = action_dim - 1

        if ignore_rotation:
            action = filter_rotation_actions(
                action, num_robots=num_robots, action_dim=action_dim
            )

    return action


def merge_actions(
    curr_action: Union[np.ndarray, List[np.ndarray]],
    prev_action: np.ndarray,
    exec_horizon: int,
) -> np.ndarray:
    """Merge previous and current actions."""
    is_list = False
    if isinstance(curr_action, list):
        is_list = True
        curr_action = np.array(curr_action)

    prev_action = utils.repeat_to_shape(curr_action, prev_action)
    prev_action = prev_action[..., :exec_horizon, :]
    curr_action = curr_action[..., :-exec_horizon, :]
    action = np.concatenate((prev_action, curr_action), axis=-2)

    return [x for x in action] if is_list else action


def subsample_actions(
    action: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    """Subsample the batch of actions."""
    assert sample_size <= action.shape[0]
    indices = np.random.choice(action.shape[0], sample_size, replace=False)
    return action[indices]

from typing import Any, Dict, Tuple, List

import torch
import numpy as np


def to_torch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def get_env_class(env_name):
    if env_name == "pusht":
        from sentinel.envs.sim_pusht.pusht_pc_env import PushTPCEnv

        return PushTPCEnv
    elif env_name == "fold_mobile":
        from sentinel.envs.sim_mobile.folding_env import FoldingEnv

        return FoldingEnv
    elif env_name == "cover_mobile":
        from sentinel.envs.sim_mobile.covering_env import CoveringEnv

        return CoveringEnv
    elif env_name == "close_mobile":
        from sentinel.envs.sim_mobile.closing_env import ClosingEnv

        return ClosingEnv
    else:
        raise ValueError()


def get_dataset(cfg, mode="train", **kwargs):
    env_name = cfg.data.dataset_class
    if env_name == "episode_dataset":
        from sentinel.bc.datasets.episode_dataset import EpisodeDataset

        return EpisodeDataset(**kwargs)

    else:
        raise ValueError()


def get_agent(agent_name):
    if agent_name == "sim3":
        from sentinel.bc.agents.sim3_agent import SIM3Agent

        return SIM3Agent
    elif agent_name == "dp":
        from sentinel.bc.agents.dp_agent import DPAgent

        return DPAgent
    elif agent_name == "sim3_dp":
        from sentinel.bc.agents.sim3_dp_agent import SIM3DPAgent

        return SIM3DPAgent
    else:
        raise ValueError(f"Agent with name [{agent_name}] not found.")


def torch_batch_to_obs(
    batch: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Convert PyTorch dataloader batch to policy observation."""
    is_batch = all(v.size(0) > 1 for v in batch.values())

    # Define key mapping.
    obs = {"depth": None}
    keymap = {"pc": "pc", "rgb": "rgb", "eef_pos": "state"}

    # Extract relevant keys.
    batch_np: Dict[str, np.ndarray] = {
        k: v.detach().cpu().numpy() for k, v in batch.items()
    }
    for k_batch, k_obs in keymap.items():
        if k_batch not in batch_np:
            continue
        # (B, H, X, Y, ...) -> (H, B, X, Y, ...)
        v = batch_np[k_batch]
        axes = list(range(v.ndim))
        axes[0], axes[1] = axes[1], axes[0]
        v = v.transpose(axes)
        if k_batch == "pc":
            v = [[pc for pc in pcs] for pcs in v]
        obs[k_obs] = v

    # Store remaining keys.
    info = {}
    for k, v in batch_np.items():
        if k in keymap:
            continue
        axes = list(range(v.ndim))
        axes[0], axes[1] = axes[1], axes[0]
        info[k] = v.transpose(axes)

    if not is_batch:
        # (H, B, N, 3) -> (H, N, 3)
        info = {k: v[:, 0] for k, v in info.items()}
        for k, v in obs.items():
            if k == "pc":
                obs[k] = [pcs[0] for pcs in v]
            else:
                obs[k] = v[:, 0] if isinstance(v, np.ndarray) else None

    return obs, info


def obs_to_batch_obs(obs: Dict[str, Any], batch_size: int = 64) -> Dict[str, Any]:
    """Convert single observation to a batch of observations."""
    batch_obs = {}
    for k in obs.keys():
        if obs[k] is None:
            batch_obs[k] = None
        elif k == "pc":
            # Point Cloud: (H, N, 3) -> (H, B, N, 3)
            assert isinstance(obs[k], list)
            batch_obs[k] = [[pc.copy()] * batch_size for pc in obs[k]]
        else:
            # Other: (H, X, Y, ...) -> (H, B, X, Y, ...)
            assert isinstance(obs[k], np.ndarray)
            data = obs[k].copy()[:, None]
            tile = [1] * data.ndim
            tile[1] = batch_size
            batch_obs[k] = np.tile(data, tile)

    return batch_obs


def batch_obs_to_list_obs(
    batch_obs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a batch of observations into a list of single observation."""
    list_obs = []
    batch_size = batch_obs["state"].shape[1]
    for b in range(batch_size):
        obs = {}
        for k in batch_obs.keys():
            if batch_obs[k] is None:
                obs[k] = None
            # Point Cloud: (H, B, N, 3) -> (H, N, 3)
            elif k == "pc":
                obs[k] = [pc[b] for pc in batch_obs[k]]
            # Other: (H, B, X, Y, ...) -> (H, X, Y, ...)
            else:
                obs[k] = batch_obs[k][:, b]
        list_obs.append(obs)

    return list_obs


def scale_action(
    action: np.ndarray,
    scale: float = 1,
    num_robots: int = 2,
    action_dim: int = 4,
    keep_dim: bool = True,
) -> np.ndarray:
    """Scale action to e.g., convert positions to velocities."""
    start_idx = 1 if action_dim > 3 else 0
    if action.ndim == 2:
        h, _ = action.shape
        action = action.reshape(h, num_robots, action_dim).copy()
        action[:, :, start_idx : start_idx + 3] *= scale
        if keep_dim:
            action = action.reshape(h, -1)
    elif action.ndim == 3:
        b, h, _ = action.shape
        action = action.reshape(b, h, num_robots, action_dim).copy()
        action[:, :, :, start_idx : start_idx + 3] *= scale
        if keep_dim:
            action = action.reshape(b, h, -1)
    else:
        raise ValueError(f"Action dimension {action.ndim} is not supported.")

    return action


def repeat_to_shape(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Repeat y to match the shape of x."""
    assert y.ndim <= x.ndim

    required_ndims = np.arange(x.ndim - y.ndim)
    y = np.expand_dims(y, axis=tuple(required_ndims))
    assert x.ndim == y.ndim

    reps = tuple(c // p for c, p in zip(x.shape, y.shape))
    y = np.tile(y, reps)
    assert x.shape == y.shape

    return y

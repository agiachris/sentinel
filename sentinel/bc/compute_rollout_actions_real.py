from typing import Any, Dict, List, Union

import re
import os
import sys
import json
import tqdm
import torch
import hydra
import wandb
import random
import pickle
import pathlib
import logging
import omegaconf
import numpy as np
import pandas as pd
import getpass as gt
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "../point_feat"))
logging.basicConfig(level=logging.ERROR)

from sentinel.bc import utils
from sentinel.utils.media import save_gif
from sentinel.bc.agents.dp_agent import DPAgent
from sentinel.bc.agents.sim3_dp_agent import SIM3DPAgent
from sentinel.bc.ood_detection.models import embedding_models
from sentinel.bc.ood_detection import action_utils


def parse_datetime(s) -> datetime:
    """Return datetime."""
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")


def parse_real_eval_mp_log(log_file: Union[str, pathlib.Path]) -> List[Dict[str, Any]]:
    """Return formatted data from real-world evaluation log."""
    # Logfile pattern match.
    pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\].*"  # Date / time.
    pattern += r"timestep (\d+) skip_steps (\d+) agent_ref_state \[(.*)\] executed_action (\[.*\]) "
    pattern += r"agent_obs_pc (\[.*\]) agent_obs_state (\[.*\]) agent_obs_rgb (\[.*\])"
    result = []
    with open(log_file, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                (
                    timestamp,
                    timestep,
                    skip_steps,
                    agent_ref_state,
                    executed_action,
                    agent_obs_pc,
                    agent_obs_state,
                    agent_obs_rgb,
                ) = match.groups()
                result.append(
                    {
                        "timestamp": parse_datetime(timestamp),
                        "timestep": int(timestep),
                        "skip_steps": int(skip_steps),
                        "agent_ref_state": np.array(
                            json.loads(agent_ref_state), dtype=np.float32
                        ),
                        "executed_action": np.array(
                            json.loads(executed_action), dtype=np.float32
                        ),
                        "agent_obs_pc": np.array(
                            json.loads(agent_obs_pc), dtype=np.float32
                        ),
                        "agent_obs_state": np.array(
                            json.loads(agent_obs_state), dtype=np.float32
                        ),
                        "agent_obs_rgb": np.array(
                            json.loads(agent_obs_rgb), dtype=np.uint8
                        ),
                    }
                )
    return result


def run_eval_real(
    cfg: omegaconf.DictConfig,
    agent: Union[DPAgent, SIM3DPAgent],
    log_dir: str,
    sample_size: int = 512,
    save_episodes: bool = True,
    save_videos: bool = True,
) -> Dict[str, Any]:
    """Evaluate agent."""
    # Make save directories.
    log_dir = pathlib.Path(log_dir)
    if (save_episodes or save_videos) and log_dir is None:
        raise ValueError("Require a log directory to save data.")
    if save_episodes:
        episode_dir = log_dir / "episodes"
        episode_dir.mkdir(exist_ok=getattr(cfg.eval, "overwrite", False))
    if save_videos:
        video_dir = log_dir / "videos"
        video_dir.mkdir(exist_ok=getattr(cfg.eval, "overwrite", False))

    # Get paths to logs.
    trial_dirs = [d for d in log_dir.iterdir() if d.is_dir() and "trial" in d.name]

    # Experiment data to save.
    save_encoder_embeddings = getattr(cfg.eval, "save_encoder_embeddings", False)
    save_resnet_embeddings = getattr(cfg.eval, "save_resnet_embeddings", False)
    save_clip_embeddings = getattr(cfg.eval, "save_clip_embeddings", False)
    save_noise_preds = getattr(cfg.eval, "save_noise_preds", False)
    save_rec_actions = getattr(cfg.eval, "save_rec_actions", False)
    if isinstance(agent, SIM3DPAgent) and (save_noise_preds or save_rec_actions):
        raise ValueError("Only supported for DPAgent.")

    # Load embedding models.
    embedding_model_device = getattr(cfg.eval, "embedding_model_device", "cpu")
    if save_resnet_embeddings:
        resnet_model = embedding_models.ResnetEmbeddingModel(
            device=embedding_model_device
        )
    if save_clip_embeddings:
        clip_model = embedding_models.ClipEmbeddingModel(device=embedding_model_device)

    # Horizons.
    obs_horizon = getattr(agent, "obs_horizon", 1)
    exec_horizon = getattr(agent, "ac_horizon", 1)
    pred_horizon = getattr(agent, "pred_horizon", 1)

    # Evaluate over episodes.
    rewards: List[float] = []
    total_timesteps = 0
    for ep_idx, trial_dir in tqdm.tqdm(enumerate(trial_dirs), desc="Episodes"):
        # Parse log data.
        log_data = parse_real_eval_mp_log(
            pathlib.Path(trial_dir) / "real_eval_mp_dp.log"
        )
        if save_episodes:
            episode_data = []
        if save_videos:
            images = []

        # Step agent (forward pass + action horizon).
        timestep = 0
        for t_data in tqdm.tqdm(log_data, desc="Timesteps"):
            assert timestep == t_data["timestep"]

            # Construct observation.
            agent_obs_pc: np.ndarray = t_data["agent_obs_pc"]
            agent_obs_state: np.ndarray = t_data["agent_obs_state"]
            assert agent_obs_pc.ndim == 3 and agent_obs_pc.shape == (
                obs_horizon,
                cfg.data.dataset.num_points,
                3,
            )
            assert agent_obs_state.ndim == 3 and agent_obs_state.shape == (
                obs_horizon,
                cfg.env.num_eef,
                cfg.env.eef_dim,
            )
            agent_obs = {"pc": [pc for pc in agent_obs_pc], "state": agent_obs_state}

            # Predict action to execute.
            executed_action = utils.scale_action(
                t_data["executed_action"].copy(),
                scale=cfg.env.args.ac_scale,
                num_robots=cfg.env.args.num_eef,
                action_dim=cfg.env.args.dof,
                keep_dim=False,
            )
            _, action_dict = agent.act(agent_obs, return_dict=True, debug=False)

            # Prediction actions to store.
            curr_batch_obs = utils.obs_to_batch_obs(agent_obs, batch_size=sample_size)
            sampled_actions, sampled_action_dicts = agent.act(
                curr_batch_obs,
                return_dict=True,
                debug=False,
            )
            sampled_actions = utils.scale_action(
                sampled_actions,
                scale=cfg.env.args.ac_scale,
                num_robots=cfg.env.args.num_eef,
                action_dim=cfg.env.args.dof,
                keep_dim=True,
            )
            assert sampled_actions.ndim == 3 and sampled_actions.shape == (
                sample_size,
                pred_horizon,
                cfg.env.args.dof,
            )

            # Store timestep in dataset.
            if save_episodes:
                result = {
                    "idx": total_timesteps,
                    "episode": ep_idx,
                    "timestep": timestep,
                    "rgb": t_data["agent_obs_rgb"],
                    "executed_action": executed_action.copy()
                    .reshape(pred_horizon, -1)
                    .astype(np.float32),
                    "sampled_actions": sampled_actions.astype(np.float32),
                    "skip_steps": t_data["skip_steps"],
                    "agent_ref_state": t_data["agent_ref_state"],
                }
                if save_encoder_embeddings:
                    assert action_dict is not None
                    if isinstance(agent, SIM3DPAgent):
                        result["encoder_feat"] = (
                            action_dict["feat_so3"][-1].flatten().astype(np.float32)
                        )
                    elif isinstance(agent, DPAgent):
                        result["encoder_feat"] = (
                            action_dict["feat"][-1].flatten().astype(np.float32)
                        )
                    else:
                        raise ValueError(
                            f"Cannot get encoder embeddings for {type(agent)}."
                        )
                    result["state_feat"] = (
                        action_dict["obs_cond_vec"].flatten().astype(np.float32)
                    )

                if save_resnet_embeddings:
                    result["resnet_feat"] = resnet_model.get_embedding(
                        t_data["agent_obs_rgb"].copy()
                    ).astype(np.float32)

                if save_clip_embeddings:
                    result["clip_feat"] = clip_model.get_embedding(
                        t_data["agent_obs_rgb"].copy()
                    ).astype(np.float32)

                if (save_rec_actions or save_noise_preds) and timestep > 0:
                    prev_batch_obs = utils.obs_to_batch_obs(
                        obs=prev_obs,
                        batch_size=sample_size,
                    )
                    merged_sampled_actions = action_utils.merge_actions(
                        curr_action=sampled_actions,
                        prev_action=prev_executed_action,
                        exec_horizon=exec_horizon,
                    )  # (B, H, D)

                if save_rec_actions:
                    # Compute per-timestep action reconstruction.
                    result["action_rec_pred"] = agent.reconstruct_action(
                        action=sampled_actions,
                        obs=curr_batch_obs,
                        return_dict=False,
                    ).astype(
                        np.float32
                    )  # (R, B, H, D)
                    result["action_rec_label"] = sampled_actions.astype(
                        np.float32
                    )  # (B, H, D)

                    # Compute temporal action reconstruction.
                    if timestep == 0:
                        result["temporal_action_rec_pred"] = result[
                            "action_rec_pred"
                        ].astype(
                            np.float32
                        )  # (R, B, H, D)
                        result["temporal_action_rec_label"] = result[
                            "action_rec_label"
                        ].astype(
                            np.float32
                        )  # (B, H, D)
                    else:
                        assert merged_sampled_actions is not None
                        result["temporal_action_rec_pred"] = (
                            agent.reconstruct_action(
                                action=merged_sampled_actions,
                                obs=prev_batch_obs,
                                return_dict=False,
                            )
                        ).astype(
                            np.float32
                        )  # (R, B, H, D)
                        result["temporal_action_rec_label"] = (
                            merged_sampled_actions  # (B, H, D)
                        ).astype(np.float32)

                if save_noise_preds:
                    # Compute per-timestep predicted errors.
                    noise_preds, noise_labels = agent.predict_noise(
                        action=sampled_actions,
                        obs=curr_batch_obs,
                        return_dict=False,
                    )
                    result["noise_preds"] = noise_preds.astype(
                        np.float32
                    )  # (S, B, H, D)
                    result["noise_labels"] = noise_labels.astype(
                        np.float32
                    )  # (S, B, H, D)

                    # Compute temporal predicted errors.
                    if timestep == 0:
                        result["temporal_noise_preds"] = result["noise_preds"].astype(
                            np.float32
                        )  # (S, B, H, D)
                        result["temporal_noise_labels"] = result["noise_labels"].astype(
                            np.float32
                        )  # (S, B, H, D)
                    else:
                        assert merged_sampled_actions is not None
                        temporal_noise_preds, temporal_noise_labels = (
                            agent.predict_noise(
                                action=merged_sampled_actions,
                                obs=prev_batch_obs,
                                return_dict=False,
                            )
                        )
                        result["temporal_noise_preds"] = temporal_noise_preds.astype(
                            np.float32
                        )  # (S, B, H, D
                        result["temporal_noise_labels"] = temporal_noise_labels.astype(
                            np.float32
                        )  # (S, B, H, D)

                episode_data.append(result)

            prev_obs = agent_obs
            prev_executed_action = executed_action.copy().reshape(pred_horizon, -1)

            if save_videos:
                images.append(t_data["agent_obs_rgb"])

            timestep += exec_horizon
            total_timesteps += 1

        rew = 1 if "success" in trial_dir.name else 0
        rewards.append(rew)

        # Save episode data.
        if save_episodes:
            episode_data = pd.DataFrame(episode_data)
            episode_data["reward"] = rew
            episode_data["success"] = rew >= 0.5
            with open(episode_dir / f"ep{ep_idx:04d}.pkl", "wb") as f:
                pickle.dump(episode_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            del episode_data

        # Save rollout gifs.
        if save_videos:
            postfix = "success" if rew >= 0.5 else "fail"
            video_path = str(video_dir / f"ep{ep_idx:04d}_{postfix}.gif")
            save_gif(np.array(images), video_path, fps=5)
            del images

    metrics = dict(rew=np.array(rewards).mean())
    return metrics


@hydra.main(config_path="configs", config_name="real_single_arm_push_chair")
def main(cfg: omegaconf.DictConfig):
    assert cfg.mode == "eval"
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Setup logging.
    if cfg.use_wandb:
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        )
        raise NotImplementedError("Set wandb username below.")
        if gt.getuser() == "<user_id>":
            user = "<user_name>"
        wandb.init(
            entity=user,
            project="sentinel",
            tags=["eval"],
            name=cfg.prefix,
            settings=wandb.Settings(code_dir="."),
            config=wandb_config,
        )

    # Load environment and agent.
    agent = utils.get_agent(cfg.agent.agent_name)(cfg)
    agent.train(False)
    agent.load_snapshot(cfg.training.ckpt)

    # Run evaluation.
    run_eval_real(
        cfg=cfg,
        agent=agent,
        log_dir=os.getcwd(),
        sample_size=cfg.training.batch_size,
        save_episodes=cfg.eval.save_episodes,
        save_videos=cfg.eval.save_videos,
    )

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

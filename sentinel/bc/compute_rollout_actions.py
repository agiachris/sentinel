from typing import Any, Dict, Optional, List, Union

import os
import sys
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

sys.path.append(os.path.join(os.path.dirname(__file__), "../point_feat"))
logging.basicConfig(level=logging.ERROR)

from sentinel.bc import utils
from sentinel.utils.media import save_gif
from sentinel.envs.sim_mobile.base_env import BaseEnv
from sentinel.envs.sim_pusht.pusht_pc_env import PushTPCEnv
from sentinel.bc.agents.dp_agent import DPAgent
from sentinel.bc.agents.sim3_dp_agent import SIM3DPAgent
from sentinel.bc.ood_detection.models import embedding_models
from sentinel.bc.ood_detection import action_utils


def organize_obs(
    render: Union[List[Any], Dict[str, Any]],
    rgb_render: Dict[str, Any],
    state: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Organize observations into dictionary."""
    if type(render) is list:
        obs = dict(
            pc=[r["pc"] for r in render],
            rgb=np.array([r["images"][0][..., :3] for r in rgb_render]),
            state=np.array(state),
        )
        for k in ["eef_pos", "eef_rot"]:
            if k in render[0]:
                obs[k] = [r[k] for r in render]
        return obs
    elif type(render) is dict:
        obs = organize_obs([render], [rgb_render], [state])
        return {k: v[0] for k, v in obs.items()}


def render_obs(env: Union[BaseEnv, PushTPCEnv]) -> Dict[str, Any]:
    """Return rendered environment observation."""
    if isinstance(env, PushTPCEnv):
        render = env.render()
    elif isinstance(env, BaseEnv):
        render = env.render(return_depth=True, return_pc=True)
    else:
        raise ValueError(f"Environment {env} not supported.")
    return render


def run_eval_sim(
    cfg: omegaconf.DictConfig,
    env: Union[BaseEnv, PushTPCEnv],
    agent: Union[DPAgent, SIM3DPAgent],
    sample_size: int = 512,
    num_episodes: int = 40,
    save_episodes: bool = True,
    save_videos: bool = True,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate agent."""
    # Make save directories.
    if (save_episodes or save_videos) and log_dir is None:
        raise ValueError("Require a log directory to save data.")
    if save_episodes:
        episode_dir = pathlib.Path(log_dir) / "episodes"
        episode_dir.mkdir(exist_ok=getattr(cfg.eval, "overwrite", False))
    if save_videos:
        video_dir = pathlib.Path(log_dir) / "videos"
        video_dir.mkdir(exist_ok=getattr(cfg.eval, "overwrite", False))

    # Experiment data to save.
    save_encoder_embeddings = getattr(cfg.eval, "save_encoder_embeddings", False)
    save_resnet_embeddings = getattr(cfg.eval, "save_resnet_embeddings", False)
    save_clip_embeddings = getattr(cfg.eval, "save_clip_embeddings", False)
    save_noise_preds = getattr(cfg.eval, "save_noise_preds", False)
    save_score_pairs = getattr(cfg.eval, "save_score_pairs", False)
    save_rec_actions = getattr(cfg.eval, "save_rec_actions", False)

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

    # Rewards.
    reward_thresh = getattr(cfg.eval, "reward_thresh", 0.75)
    term_on_thresh = getattr(cfg.eval, "term_on_thresh", False)

    # Evaluate over episodes.
    rewards: List[float] = []
    total_timesteps = 0
    for ep_idx in tqdm.tqdm(range(num_episodes), desc="Episodes"):
        # Initial state and observation.
        state = env.reset()
        rgb_render = render = render_obs(env)
        if save_videos:
            images = [rgb_render["images"][0][..., :3]]
        obs = organize_obs(render, rgb_render, state)
        obs_history = [obs] * obs_horizon

        if save_episodes:
            episode_data = []

        # Step agent (forward pass + action horizon).
        done = False
        prev_reward = None
        timestep = 0
        with tqdm.tqdm(
            total=env.max_episode_length, desc="Timesteps", leave=False
        ) as pbar:
            while not done:
                # Construct observation.
                agent_obs = {}
                for k in obs.keys():
                    agent_obs[k] = [o[k] for o in obs_history[-obs_horizon:]]
                    if k != "pc":
                        agent_obs[k] = np.stack(agent_obs[k])

                # Predict action to execute.
                executed_action, action_dict = agent.act(
                    agent_obs, return_dict=True, debug=False
                )
                if isinstance(env, PushTPCEnv):
                    keep_dim = True
                elif isinstance(env, BaseEnv):
                    keep_dim = False
                else:
                    raise ValueError(f"Environment {env} not supported.")
                executed_action = utils.scale_action(
                    executed_action,
                    scale=env.args.ac_scale,
                    num_robots=env.args.num_eef,
                    action_dim=env.args.dof,
                    keep_dim=keep_dim,
                )
                if (
                    len(agent_obs["pc"]) == 0
                    or len(agent_obs["pc"][0]) == 0
                    or action_dict is None
                ):
                    action_dict = None
                    break

                # Prediction actions to store.
                curr_batch_obs = utils.obs_to_batch_obs(
                    agent_obs, batch_size=sample_size
                )
                sampled_actions, sampled_action_dicts = agent.act(
                    curr_batch_obs,
                    return_dict=True,
                    debug=False,
                )
                sampled_actions = utils.scale_action(
                    sampled_actions,
                    scale=env.args.ac_scale,
                    num_robots=env.args.num_eef,
                    action_dim=env.args.dof,
                    keep_dim=True,
                )

                # Store timestep in dataset.
                if save_episodes:
                    result = {
                        "idx": total_timesteps,
                        "episode": ep_idx,
                        "timestep": timestep,
                        "rgb": agent_obs["rgb"][-1].astype(np.uint8),
                        "executed_action": executed_action.copy()
                        .reshape(pred_horizon, -1)
                        .astype(np.float32),
                        "sampled_actions": sampled_actions.astype(np.float32),
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
                            agent_obs["rgb"][-1]
                        ).astype(np.float32)

                    if save_clip_embeddings:
                        result["clip_feat"] = clip_model.get_embedding(
                            agent_obs["rgb"][-1]
                        ).astype(np.float32)

                    if (
                        save_rec_actions or save_noise_preds or save_score_pairs
                    ) and timestep > 0:
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
                            result["temporal_noise_preds"] = result[
                                "noise_preds"
                            ].astype(
                                np.float32
                            )  # (S, B, H, D)
                            result["temporal_noise_labels"] = result[
                                "noise_labels"
                            ].astype(
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
                            result["temporal_noise_preds"] = (
                                temporal_noise_preds.astype(np.float32)
                            )  # (S, B, H, D
                            result["temporal_noise_labels"] = (
                                temporal_noise_labels.astype(np.float32)
                            )  # (S, B, H, D)

                    if save_score_pairs:
                        result["curr_noise_scores"] = (
                            np.stack([d["noise_preds"] for d in sampled_action_dicts])
                            .transpose((1, 0, 2, 3))[..., :-exec_horizon, :]
                            .astype(np.float32)
                        )  # (T, B, H, D)

                        # Compute noise score pairs.
                        if timestep == 0:
                            result["prev_noise_scores"] = result[
                                "curr_noise_scores"
                            ].astype(np.float32)
                        else:
                            merged_diff_actions = action_utils.merge_actions(
                                curr_action=np.stack(
                                    [d["diff_actions"] for d in sampled_action_dicts]
                                ),
                                prev_action=prev_action_dict["diff_actions"],
                                exec_horizon=exec_horizon,
                            ).transpose(
                                (1, 0, 2, 3)
                            )  # (T, B, H, D)
                            result["prev_noise_scores"] = agent.reconstruct_noise(
                                diff_action=merged_diff_actions,
                                obs=prev_batch_obs,
                                return_dict=False,
                            )[..., exec_horizon:, :].astype(
                                np.float32
                            )  # (T, B, H, D)

                    episode_data.append(result)

                prev_obs = agent_obs
                prev_executed_action = executed_action.copy().reshape(pred_horizon, -1)
                prev_action_dict = action_dict

                # Execute actions.
                for ac_idx in range(exec_horizon):
                    if len(obs["pc"]) == 0 or len(obs["pc"][0]) == 0:
                        action_dict = None
                        break

                    # Take step.
                    agent_ac = (
                        executed_action[ac_idx]
                        if len(executed_action.shape) > 1
                        else executed_action
                    )
                    state, rew, done, _ = env.step(agent_ac, dummy_reward=True)

                    # Store observation.
                    rgb_render = render = render_obs(env)
                    if save_videos:
                        images.append(rgb_render["images"][0][..., :3])
                    obs = organize_obs(render, rgb_render, state)
                    obs_history.append(obs)
                    if len(obs_history) > obs_horizon:
                        obs_history = obs_history[-obs_horizon:]

                    curr_reward = env.compute_reward()
                    if prev_reward is None or curr_reward > prev_reward:
                        prev_reward = curr_reward

                    if term_on_thresh and prev_reward >= reward_thresh:
                        done = True
                        break

                    # Increment timesteps.
                    timestep += 1
                    total_timesteps += 1
                    pbar.update(1)
                    pbar.set_postfix({"Reward": prev_reward})

                    if (
                        action_dict is None
                        or done
                        or getattr(env, "_failed_safety_check", False)
                    ):
                        break

                if (
                    action_dict is None
                    or done
                    or getattr(env, "_failed_safety_check", False)
                ):
                    break

        rew = prev_reward if action_dict is None else env.compute_reward()
        rewards.append(rew)

        # Save episode data.
        if save_episodes:
            episode_data = pd.DataFrame(episode_data)
            episode_data["reward"] = rew
            episode_data["success"] = rew >= reward_thresh
            with open(episode_dir / f"ep{ep_idx:04d}.pkl", "wb") as f:
                pickle.dump(episode_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            del episode_data

        # Save rollout gifs.
        if save_videos:
            postfix = "success" if rew >= reward_thresh else "fail"
            video_path = str(video_dir / f"ep{ep_idx:04d}_{postfix}.gif")
            save_gif(np.array(images), video_path, fps=5)
            del images

        # Avoid memory leak.
        if isinstance(env, BaseEnv):
            del env._frames

    metrics = dict(rew=np.array(rewards).mean())
    return metrics


@hydra.main(config_path="configs", config_name="close_mobile_dp")
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
    env = utils.get_env_class(cfg.env.env_class)(cfg.env.args)
    agent = utils.get_agent(cfg.agent.agent_name)(cfg)
    agent.train(False)
    agent.load_snapshot(cfg.training.ckpt)

    # Run evaluation.
    eval_metrics = run_eval_sim(
        cfg=cfg,
        env=env,
        agent=agent,
        sample_size=cfg.training.batch_size,
        num_episodes=cfg.eval.num_eval_episodes,
        save_episodes=cfg.eval.save_episodes,
        save_videos=cfg.eval.save_videos,
        log_dir=os.getcwd(),
    )

    print("Evaluation results:")
    print(eval_metrics)
    if cfg.use_wandb:
        wandb.log({"eval/" + k: v for k, v in eval_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main()

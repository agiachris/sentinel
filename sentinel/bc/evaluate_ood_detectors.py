from typing import Union, Dict, Callable, Any, List, Optional

import os
import sys
import torch
import hydra
import random
import pickle
import imageio
import pathlib
import omegaconf
import numpy as np
import pandas as pd
from collections import defaultdict

# Set the logging level for httpx to WARNING or ERROR to suppress INFO messages/
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

sys.path.append(os.path.join(os.path.dirname(__file__), "../point_feat"))

from sentinel.bc import utils
import sentinel.bc.datasets.utils as data_utils
from sentinel.bc.ood_detection import error_utils, metric_utils, action_utils
from sentinel.bc.datasets.episode_dataset import EpisodeDataset, VideoDataset
from sentinel.bc.ood_detection.prompt_utils import PromptManager
from sentinel.bc.ood_detection.models import get_vlm_cls, VisionLanguageModel

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="hydra._internal.defaults_list"
)
warnings.filterwarnings("ignore", category=UserWarning, module="hydra._internal.hydra")


DEBUG = 0
BREAKPOINT = 0
dbprint = print if DEBUG == 1 else lambda *args: ()


# Evaluation macros.
CONSISTENCY_AGGR_FNS: Dict[str, Callable[[np.ndarray], float]] = {
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
    "std_dev": np.std,
    "var": np.var,
}


CONSISTENCY_ERROR_FNS: Dict[str, Dict[str, Any]] = {
    "mse_all": {
        "error_fn": "mse",
        "ignore_gripper": True,
        "ignore_rotation": False,
    },
    "mse_pos": {
        "error_fn": "mse",
        "ignore_gripper": True,
        "ignore_rotation": True,
    },
    "ate_pos": {
        "error_fn": "ate",
        "ignore_gripper": True,
        "ignore_rotation": True,
    },
}


CONSISTENCY_DIST_ERROR_FNS = {
    # MMD.
    "mmd_rbf_pos": {
        "error_fn": "mmd_rbf",
        "ignore_gripper": True,
        "ignore_rotation": True,
        "gamma": 1.0,
    },
    "mmd_rbf_all": {
        "error_fn": "mmd_rbf",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "gamma": None,
    },
    "mmd_rbf_all_1.0": {
        "error_fn": "mmd_rbf",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "gamma": 1.0,
    },
    "mmd_rbf_all_median": {
        "error_fn": "mmd_rbf",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "gamma": "median",
    },
    "mmd_rbf_all_eig": {
        "error_fn": "mmd_rbf",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "gamma": "max_eig",
    },
    "mmd_rbf_all_0.1": {
        "error_fn": "mmd_rbf",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "gamma": 0.1,
    },
    "mmd_rbf_all_0.5": {
        "error_fn": "mmd_rbf",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "gamma": 0.5,
    },
    "mmd_rbf_all_5.0": {
        "error_fn": "mmd_rbf",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "gamma": 5.0,
    },
    "mmd_rbf_all_10.0": {
        "error_fn": "mmd_rbf",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "gamma": 10.0,
    },
    "mmd_rbf_all_100.0": {
        "error_fn": "mmd_rbf",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "gamma": 100.0,
    },
    # KDE For. KL.
    "kde_kl_all_for": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": True,
        "bandwidth": 1.0,
    },
    "kde_kl_all_for_eig": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": True,
        "bandwidth": "max_eig",
    },
    "kde_kl_all_for_0.1": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": True,
        "bandwidth": 0.1,
    },
    "kde_kl_all_for_0.5": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": True,
        "bandwidth": 0.5,
    },
    "kde_kl_all_for_5.0": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": True,
        "bandwidth": 5.0,
    },
    "kde_kl_all_for_10.0": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": True,
        "bandwidth": 10.0,
    },
    "kde_kl_all_for_100.0": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": True,
        "bandwidth": 100.0,
    },
    # KDE Rev. KL.
    "kde_kl_all_rev": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": False,
        "bandwidth": 1.0,
    },
    "kde_kl_all_rev_eig": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": False,
        "bandwidth": "max_eig",
    },
    "kde_kl_all_rev_0.1": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": False,
        "bandwidth": 0.1,
    },
    "kde_kl_all_rev_0.5": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": False,
        "bandwidth": 0.5,
    },
    "kde_kl_all_rev_5.0": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": False,
        "bandwidth": 5.0,
    },
    "kde_kl_all_rev_10.0": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": False,
        "bandwidth": 10.0,
    },
    "kde_kl_all_rev_100.0": {
        "error_fn": "kde_kl",
        "ignore_gripper": True,
        "ignore_rotation": False,
        "forward": False,
        "bandwidth": 100.0,
    },
}


ENSEMBLE_ACTION_SPACES: Dict[str, Dict[str, Any]] = {
    "all": {
        "ignore_gripper": True,
        "ignore_rotation": False,
    },
    "pos": {
        "ignore_gripper": True,
        "ignore_rotation": True,
    },
    "traj": {"ignore_gripper": True, "ignore_rotation": True, "use_trajectory": True},
}


EMBEDDING_SCORE_FNS: Dict[str, Dict[str, Any]] = {
    "top1_l2": {"method": "topk", "method_kwargs": {"error_fn": "l2", "k": 1}},
    "top5_l2": {"method": "topk", "method_kwargs": {"error_fn": "l2", "k": 5}},
    "top10_l2": {"method": "topk", "method_kwargs": {"error_fn": "l2", "k": 10}},
    "top1_cosine": {"method": "topk", "method_kwargs": {"error_fn": "cosine", "k": 1}},
    "top5_cosine": {"method": "topk", "method_kwargs": {"error_fn": "cosine", "k": 5}},
    "top10_cosine": {
        "method": "topk",
        "method_kwargs": {"error_fn": "cosine", "k": 10},
    },
    "mahal": {
        "method": "mahal",
    },
}


LOSS_FN_TENSORS = {
    "score_matching": {
        "pred": "prev_noise_scores",
        "label": "curr_noise_scores",
    },
    "noise_pred": {
        "pred": "noise_preds",
        "label": "noise_labels",
    },
    "temporal_noise_pred": {
        "pred": "temporal_noise_preds",
        "label": "temporal_noise_labels",
    },
    "action_rec": {
        "pred": "action_rec_pred",
        "label": "action_rec_label",
    },
    "temporal_action_rec": {
        "pred": "temporal_action_rec_pred",
        "label": "temporal_action_rec_label",
    },
}


LOSS_FNS = {
    "score_matching_all": {
        "sample_size_key": "score_matching",
        "loss_fn_key": "score_matching",
        "loss_fn_kwargs": {
            "ignore_gripper": True,
            "ignore_rotation": False,
        },
    },
    "score_matching_pos": {
        "sample_size_key": "score_matching",
        "loss_fn_key": "score_matching",
        "loss_fn_kwargs": {
            "ignore_gripper": True,
            "ignore_rotation": True,
        },
    },
    "noise_pred_all": {
        "sample_size_key": "noise_pred",
        "loss_fn_key": "noise_pred",
        "loss_fn_kwargs": {
            "ignore_gripper": True,
            "ignore_rotation": False,
        },
    },
    "temporal_noise_pred_all": {
        "sample_size_key": "noise_pred",
        "loss_fn_key": "temporal_noise_pred",
        "loss_fn_kwargs": {
            "ignore_gripper": True,
            "ignore_rotation": False,
        },
    },
    "action_rec_all": {
        "sample_size_key": "action_rec",
        "loss_fn_key": "action_rec",
        "loss_fn_kwargs": {
            "ignore_gripper": True,
            "ignore_rotation": False,
        },
    },
    "temporal_action_rec_all": {
        "sample_size_key": "action_rec",
        "loss_fn_key": "temporal_action_rec",
        "loss_fn_kwargs": {
            "ignore_gripper": True,
            "ignore_rotation": False,
        },
    },
}


# Experiment keys.
def temporal_consistency_exp_key(
    pred_horizon: int,
    sample_size: int,
    error_fn: str,
    aggr_fn: Optional[str] = None,
) -> str:
    if error_fn in CONSISTENCY_DIST_ERROR_FNS:
        return (
            f"pred_horizon_{pred_horizon}_sample_size_{sample_size}_error_fn_{error_fn}"
        )
    else:
        return f"pred_horizon_{pred_horizon}_sample_size_{sample_size}_error_fn_{error_fn}_aggr_fn_{aggr_fn}"


def vlm_exp_key(
    model: str,
    template: str,
) -> str:
    return f"model_{model}_template_{template}"


def diffusion_ensemble_exp_key(
    pred_horizon: int,
    sample_size: int,
    action_space: str,
) -> str:
    return f"pred_horizon_{pred_horizon}_sample_size_{sample_size}_action_space_{action_space}"


def embedding_similarity_exp_key(
    embedding: str,
    score_fn: int,
) -> str:
    return f"embedding_{embedding}_score_fn_{score_fn}"


def loss_function_exp_key(
    loss_fn: str,
    sample_size: int,
) -> str:
    return f"loss_fn_{loss_fn}_sample_size_{sample_size}"


def quantile_exp_key(exp_key: str, quantile: float = 0.95) -> str:
    return f"{exp_key}_quantile_{quantile}"


def get_consistency_aggr_fns(
    cfg: omegaconf.DictConfig,
    error_fn: str,
) -> List[Optional[str]]:
    if error_fn not in CONSISTENCY_ERROR_FNS:
        return [None]
    return cfg.eval.consistency.aggr_fns


# Dataframe utils.
def compute_cum_scores(
    results_frame: pd.DataFrame,
    exp_keys: List[str],
) -> pd.DataFrame:
    for exp_key in exp_keys:
        cum_scores = pd.Series(
            data_utils.aggr_episode_key_data(
                results_frame,
                f"{exp_key}_score",
                np.cumsum,
            ),
            name=f"{exp_key}_cum_score",
        )
        results_frame = pd.concat([results_frame, cum_scores], axis=1)

    return results_frame


def get_rgb(data: Dict[str, Any]) -> Optional[np.ndarray]:
    d: Dict[str, Any] = data.get("obs", data)
    rgb = d.get("rgb", None)
    if isinstance(rgb, np.ndarray):
        assert rgb.ndim in [3, 4]
        rgb = rgb[-1] if rgb.ndim == 4 else rgb
    return rgb


def compute_loss_function_scores(
    cfg: omegaconf.DictConfig,
    dataset: EpisodeDataset,
) -> pd.DataFrame:
    """Compute loss function scores over dataset."""
    results = []
    exp_keys = []

    for data in iter(dataset):

        results.append(
            {
                "episode": data["episode"],
                "timestep": data["timestep"],
                "success": data.get("success", True),
            }
        )
        rgb = get_rgb(data)
        if isinstance(rgb, np.ndarray):
            results[-1]["rgb"] = rgb

        for loss_fn in cfg.eval.loss_functions.loss_fns:
            loss_fn_key = LOSS_FNS[loss_fn]["loss_fn_key"]
            loss_fn_kwargs = LOSS_FNS[loss_fn]["loss_fn_kwargs"]
            pred_key = LOSS_FN_TENSORS[loss_fn_key]["pred"]
            label_key = LOSS_FN_TENSORS[loss_fn_key]["label"]
            sample_size_key = LOSS_FNS[loss_fn]["sample_size_key"]

            for sample_size in getattr(
                cfg.eval.loss_functions, f"{sample_size_key}_sample_sizes"
            ):

                exp_key = loss_function_exp_key(
                    loss_fn=loss_fn,
                    sample_size=sample_size,
                )
                if exp_key not in exp_keys:
                    exp_keys.append(exp_key)

                pred: np.ndarray = data[pred_key]
                label: np.ndarray = data[label_key]

                if label.shape != pred.shape:
                    label = utils.repeat_to_shape(pred, label)

                # Compute sample loss.
                loss = (pred - label) ** 2
                if loss_fn_kwargs is not None:
                    assert isinstance(loss_fn_kwargs, dict)
                    if cfg.env.dof >= 4:
                        loss = action_utils.filter_actions(
                            action=loss,
                            num_robots=cfg.env.num_eef,
                            action_dim=cfg.env.dof,
                            ignore_gripper=loss_fn_kwargs.get("ignore_gripper", True),
                            ignore_rotation=loss_fn_kwargs.get("ignore_rotation", True),
                        )

                assert sample_size <= loss.shape[0]  # (S, B, H, D)
                loss = loss[:sample_size].mean().item()

                results[-1][f"{exp_key}_score"] = loss

    results_frame = compute_cum_scores(pd.DataFrame(results), exp_keys)
    return results_frame


def evaluate_loss_functions(
    cfg: omegaconf.DictConfig,
    demo_dataset_path: Union[str, pathlib.Path],
    test_dataset_path: Union[str, pathlib.Path],
) -> Dict[str, Union[Dict[str, Any], pd.DataFrame]]:
    """Compute loss function results."""
    # Construct episode iterable datasets.
    demo_dataset = EpisodeDataset(
        dataset_path=demo_dataset_path,
        exec_horizon=1 if cfg.eval.calib_on_light else cfg.model.ac_horizon,
        sample_history=0,
        filter_success=getattr(cfg.eval, "filter_demo_success", False),
        filter_failure=getattr(cfg.eval, "filter_demo_failure", True),
        filter_episodes=getattr(cfg.eval, "filter_demo_episodes", None),
        max_episode_length=getattr(cfg.eval, "max_demo_episode_length", None),
        max_num_episodes=getattr(cfg.eval, "max_num_episodes", None),
    )
    test_dataset = EpisodeDataset(
        dataset_path=test_dataset_path,
        exec_horizon=1,
        sample_history=0,
        filter_success=getattr(cfg.eval, "filter_test_success", False),
        filter_failure=getattr(cfg.eval, "filter_test_failure", False),
        filter_episodes=getattr(cfg.eval, "filter_test_episodes", None),
        max_episode_length=getattr(cfg.eval, "max_test_episode_length", None),
        max_num_episodes=getattr(cfg.eval, "max_num_episodes", None),
    )

    # Compute scores for specified parameter sets.
    results_dict = defaultdict(dict)
    demo_results_frame = compute_loss_function_scores(cfg, demo_dataset)
    test_results_frame = compute_loss_function_scores(cfg, test_dataset)

    # Compute metrics for specified parameter sets.
    for loss_fn in cfg.eval.loss_functions.loss_fns:
        sample_size_key = LOSS_FNS[loss_fn]["sample_size_key"]
        for sample_size in getattr(
            cfg.eval.loss_functions, f"{sample_size_key}_sample_sizes"
        ):

            exp_key = loss_function_exp_key(
                loss_fn=loss_fn,
                sample_size=sample_size,
            )

            for quantile in cfg.eval.quantiles:

                test_results_frame = metric_utils.compute_detection_results(
                    exp_key=exp_key,
                    quantile_key=quantile_exp_key(exp_key, quantile),
                    results_dict=results_dict,
                    demo_results_frame=demo_results_frame,
                    test_results_frame=test_results_frame,
                    detector=getattr(cfg.eval, "detector", "quantile"),
                    detector_kwargs={
                        "quantile": quantile,
                        **getattr(cfg.eval, "detector_kwargs", {}),
                    },
                )

    return {
        "results_dict": results_dict,
        "test_results_frame": test_results_frame,
        "demo_results_frame": demo_results_frame,
    }


def compute_embedding_similarity_scores(
    cfg: omegaconf.DictConfig,
    test_dataset: Optional[EpisodeDataset] = None,
    demo_dataset: Optional[EpisodeDataset] = None,
    demo_frame: Optional[pd.DataFrame] = None,
    leave_timestep_out: bool = False,
    leave_episode_out: bool = False,
    demo_as_test: bool = False,
) -> pd.DataFrame:
    """Compute embedding similarity scores over dataset."""
    assert not (leave_timestep_out and leave_episode_out)
    assert (demo_dataset is not None) ^ (demo_frame is not None)
    assert (test_dataset is not None) ^ demo_as_test

    # Extract demo embeddings.
    if demo_dataset is not None:
        demo_frame = []
        for data in iter(demo_dataset):
            demo_frame.append(
                {
                    "episode": data["episode"],
                    "timestep": data["timestep"],
                    "success": data.get("success", True),
                }
            )
            rgb = get_rgb(data)
            if isinstance(rgb, np.ndarray):
                demo_frame[-1]["rgb"] = rgb

            for embedding in cfg.eval.embedding.embeddings:
                demo_frame[-1][embedding] = data[embedding].flatten()

        demo_frame = pd.DataFrame(demo_frame)
    assert isinstance(demo_frame, pd.DataFrame)

    # Extract test embeddings.
    if demo_as_test:
        test_frame = demo_frame.copy()
    else:
        test_frame = []
        for data in iter(test_dataset):
            test_frame.append(
                {
                    "episode": data["episode"],
                    "timestep": data["timestep"],
                    "success": data.get("success", True),
                }
            )
            rgb = get_rgb(data)
            if isinstance(rgb, np.ndarray):
                test_frame[-1]["rgb"] = rgb

            for embedding in cfg.eval.embedding.embeddings:
                test_frame[-1][embedding] = data[embedding].flatten()

        test_frame = pd.DataFrame(test_frame)
    assert isinstance(test_frame, pd.DataFrame)

    # Compute embedding scores.
    exp_keys = []
    for embedding in cfg.eval.embedding.embeddings:
        for score_fn in cfg.eval.embedding.score_fns:

            exp_key = embedding_similarity_exp_key(
                embedding=embedding,
                score_fn=score_fn,
            )
            if exp_key not in exp_keys:
                exp_keys.append(exp_key)

            if leave_episode_out:
                test_frame = pd.concat(
                    [
                        test_frame,
                        pd.Series(np.zeros(len(test_frame)), name=f"{exp_key}_score"),
                    ],
                    axis=1,
                )
                for i in range(data_utils.num_episodes(test_frame)):
                    episode_frame = data_utils.get_episode(
                        test_frame, i, use_index=True
                    )
                    episode = episode_frame.iloc[0].to_dict()["episode"]
                    non_episode_frame: pd.DataFrame = demo_frame[
                        demo_frame["episode"] != episode
                    ]

                    episode_scores = error_utils.compute_embedding_scores(
                        data_embeddings=non_episode_frame[embedding].values,
                        test_embeddings=episode_frame[embedding].values,
                        **EMBEDDING_SCORE_FNS[score_fn],
                    )
                    test_frame.loc[
                        test_frame["episode"] == episode, f"{exp_key}_score"
                    ] = episode_scores
            else:
                test_scores = error_utils.compute_embedding_scores(
                    data_embeddings=demo_frame[embedding].values,
                    test_embeddings=test_frame[embedding].values,
                    leave_one_out=leave_timestep_out,
                    **EMBEDDING_SCORE_FNS[score_fn],
                )
                test_frame = pd.concat(
                    [test_frame, pd.Series(test_scores, name=f"{exp_key}_score")],
                    axis=1,
                )

    test_frame = compute_cum_scores(test_frame, exp_keys)
    return test_frame


def evaluate_embedding_similarity(
    cfg: omegaconf.DictConfig,
    demo_dataset_path: Union[str, pathlib.Path],
    test_dataset_path: Union[str, pathlib.Path],
) -> Dict[str, Union[Dict[str, Any], pd.DataFrame]]:
    """Compute embedding similarity results."""
    # Construct episode iterable datasets.
    demo_dataset = EpisodeDataset(
        dataset_path=demo_dataset_path,
        exec_horizon=1 if cfg.eval.calib_on_light else cfg.model.ac_horizon,
        sample_history=0,
        filter_success=getattr(cfg.eval, "filter_demo_success", False),
        filter_failure=getattr(cfg.eval, "filter_demo_failure", True),
        filter_episodes=getattr(cfg.eval, "filter_demo_episodes", None),
        max_episode_length=getattr(cfg.eval, "max_demo_episode_length", None),
        max_num_episodes=getattr(cfg.eval, "max_num_episodes", None),
    )
    test_dataset = EpisodeDataset(
        dataset_path=test_dataset_path,
        exec_horizon=1,
        sample_history=0,
        filter_success=getattr(cfg.eval, "filter_test_success", False),
        filter_failure=getattr(cfg.eval, "filter_test_failure", False),
        filter_episodes=getattr(cfg.eval, "filter_test_episodes", None),
        max_episode_length=getattr(cfg.eval, "max_test_episode_length", None),
        max_num_episodes=getattr(cfg.eval, "max_num_episodes", None),
    )

    # Compute scores for specified parameter sets.
    results_dict = defaultdict(dict)
    demo_results_frame = compute_embedding_similarity_scores(
        cfg,
        demo_dataset=demo_dataset,
        demo_as_test=True,
        leave_episode_out=getattr(cfg.eval.embedding, "leave_episode_out", True),
        leave_timestep_out=getattr(cfg.eval.embedding, "leave_timestep_out", False),
    )
    test_results_frame = compute_embedding_similarity_scores(
        cfg,
        test_dataset=test_dataset,
        demo_frame=demo_results_frame,
    )

    # Compute metrics for specified parameter sets.
    for embedding in cfg.eval.embedding.embeddings:
        for score_fn in cfg.eval.embedding.score_fns:

            exp_key = embedding_similarity_exp_key(
                embedding=embedding,
                score_fn=score_fn,
            )

            for quantile in cfg.eval.quantiles:

                test_results_frame = metric_utils.compute_detection_results(
                    exp_key=exp_key,
                    quantile_key=quantile_exp_key(exp_key, quantile),
                    results_dict=results_dict,
                    demo_results_frame=demo_results_frame,
                    test_results_frame=test_results_frame,
                    detector=getattr(cfg.eval, "detector", "quantile"),
                    detector_kwargs={
                        "quantile": quantile,
                        **getattr(cfg.eval, "detector_kwargs", {}),
                    },
                )

    return {
        "results_dict": results_dict,
        "test_results_frame": test_results_frame,
        "demo_results_frame": demo_results_frame,
    }


def compute_diffusion_ensemble_variances(
    cfg: omegaconf.DictConfig,
    dataset: EpisodeDataset,
) -> pd.DataFrame:
    """Compute diffusion ensemble variances over dataset."""
    results = []
    exp_keys = []

    for data in iter(dataset):

        results.append(
            {
                "episode": data["episode"],
                "timestep": data["timestep"],
                "success": data.get("success", True),
            }
        )
        rgb = get_rgb(data)
        if isinstance(rgb, np.ndarray):
            results[-1]["rgb"] = rgb

        for sample_size in cfg.eval.ensemble.sample_sizes:

            # Subsample current actions.
            actions = action_utils.subsample_actions(
                data["sampled_actions"],
                sample_size,
            )

            for pred_horizon in cfg.eval.ensemble.pred_horizons:
                for action_space in cfg.eval.ensemble.action_spaces:

                    exp_key = diffusion_ensemble_exp_key(
                        pred_horizon=pred_horizon,
                        sample_size=sample_size,
                        action_space=action_space,
                    )
                    if exp_key not in exp_keys:
                        exp_keys.append(exp_key)

                    # Compute variance (vectorized).
                    variance = error_utils.compute_action_variance(
                        actions=actions,
                        pred_horizon=pred_horizon,
                        sim_freq=cfg.env.args.freq,
                        num_robots=cfg.env.num_eef,
                        action_dim=cfg.env.dof,
                        **ENSEMBLE_ACTION_SPACES[action_space],
                    )
                    results[-1][f"{exp_key}_score"] = variance

    results_frame = compute_cum_scores(pd.DataFrame(results), exp_keys)
    return results_frame


def evaluate_diffusion_ensemble(
    cfg: omegaconf.DictConfig,
    demo_dataset_path: Union[str, pathlib.Path],
    test_dataset_path: Union[str, pathlib.Path],
) -> Dict[str, Union[Dict[str, Any], pd.DataFrame]]:
    """Compute diffusion ensemble results."""
    # Construct episode iterable datasets.
    demo_dataset = EpisodeDataset(
        dataset_path=demo_dataset_path,
        exec_horizon=1 if cfg.eval.calib_on_light else cfg.model.ac_horizon,
        sample_history=0,
        filter_success=getattr(cfg.eval, "filter_demo_success", False),
        filter_failure=getattr(cfg.eval, "filter_demo_failure", True),
        filter_episodes=getattr(cfg.eval, "filter_demo_episodes", None),
        max_episode_length=getattr(cfg.eval, "max_demo_episode_length", None),
        max_num_episodes=getattr(cfg.eval, "max_num_episodes", None),
    )
    test_dataset = EpisodeDataset(
        dataset_path=test_dataset_path,
        exec_horizon=1,
        sample_history=0,
        filter_success=getattr(cfg.eval, "filter_test_success", False),
        filter_failure=getattr(cfg.eval, "filter_test_failure", False),
        filter_episodes=getattr(cfg.eval, "filter_test_episodes", None),
        max_episode_length=getattr(cfg.eval, "max_test_episode_length", None),
        max_num_episodes=getattr(cfg.eval, "max_num_episodes", None),
    )

    # Compute scores for specified parameter sets.
    results_dict = defaultdict(dict)
    demo_results_frame = compute_diffusion_ensemble_variances(cfg, demo_dataset)
    test_results_frame = compute_diffusion_ensemble_variances(cfg, test_dataset)

    # Compute metrics for specified parameter sets.
    for sample_size in cfg.eval.ensemble.sample_sizes:
        for pred_horizon in cfg.eval.ensemble.pred_horizons:
            for action_space in cfg.eval.ensemble.action_spaces:

                exp_key = diffusion_ensemble_exp_key(
                    pred_horizon=pred_horizon,
                    sample_size=sample_size,
                    action_space=action_space,
                )

                for quantile in cfg.eval.quantiles:

                    test_results_frame = metric_utils.compute_detection_results(
                        exp_key=exp_key,
                        quantile_key=quantile_exp_key(exp_key, quantile),
                        results_dict=results_dict,
                        demo_results_frame=demo_results_frame,
                        test_results_frame=test_results_frame,
                        detector=getattr(cfg.eval, "detector", "quantile"),
                        detector_kwargs={
                            "quantile": quantile,
                            **getattr(cfg.eval, "detector_kwargs", {}),
                        },
                    )

    return {
        "results_dict": results_dict,
        "test_results_frame": test_results_frame,
        "demo_results_frame": demo_results_frame,
    }


def create_gif(
    images: List[np.ndarray],
    filename: str,
) -> None:
    filepath = pathlib.Path(os.path.dirname(__file__)) / f"{filename}.gif"
    imageio.mimsave(filepath, images, fps=1, loop=0)


# Evaluate scripts.
def compute_vlm_predictions(
    cfg: omegaconf.DictConfig,
    ref_dataset: VideoDataset,
    test_dataset: VideoDataset,
) -> pd.DataFrame:
    """Compute vision-language model predictions over dataset."""
    results = []

    def episode_to_images(episode: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Extract image frames from episode."""
        images = [get_rgb(x) for x in episode]
        assert all(isinstance(x, np.ndarray) and x.ndim == 3 for x in images)
        return images

    if DEBUG:
        reference_images = None
        goal_images = None

    # Prepare models and prompt managers.
    models: Dict[str, VisionLanguageModel] = {}
    prompts: Dict[str, PromptManager] = {}
    for model in cfg.eval.vlm.models:
        # Store model.
        models[model] = get_vlm_cls(model)(model=model)

        # Store template.
        for template in getattr(cfg.eval.vlm.templates, model):
            if template not in prompts:
                prompt_manager = PromptManager(
                    template=pathlib.Path(cfg.eval.vlm.prompt_dir) / f"{template}.yaml",
                    domain=cfg.eval.vlm.domain,
                    crop_images=True,
                    resize_images=True,
                    subsample_freq=test_dataset._subsample_freq,
                )

                # Store reference video.
                if prompt_manager.settings.get("use_reference", False):
                    reference_episode_idx = prompt_manager.reference_episode
                    reference_episode = ref_dataset.get_episode(reference_episode_idx)
                    reference_images = episode_to_images(reference_episode)
                    prompt_manager.reference = reference_images
                    del reference_episode

                # Store reference goals.
                if prompt_manager.settings.get("use_goals", False):
                    goal_images = []
                    for goal_episode_idx in prompt_manager.goal_episodes:
                        goal_episode = ref_dataset.get_episode(goal_episode_idx)
                        goal_images.append(episode_to_images(goal_episode)[-1].copy())
                        del goal_episode
                    prompt_manager.goals = goal_images

                prompts[template] = prompt_manager

    if DEBUG:
        if reference_images is not None:
            create_gif(reference_images, "reference")
        if goal_images is not None:
            create_gif(goal_images, "goals")

    # Compute VLM predictions.
    for data in iter(test_dataset):
        results.append(
            {
                "episode": data[-1]["episode"],
                "timestep": data[-1]["timestep"],
                "success": data[-1].get("success", True),
            }
        )
        rgb = get_rgb(data[-1])
        if isinstance(rgb, np.ndarray):
            results[-1]["rgb"] = rgb

        curr_images = episode_to_images(data)
        if DEBUG:
            create_gif(curr_images, "current")
        for model in cfg.eval.vlm.models:
            for template in getattr(cfg.eval.vlm.templates, model):
                exp_key = vlm_exp_key(model=model, template=template)

                # Construct prompt.
                messages, client_kwargs = prompts[template].construct_prompt(
                    model=model,
                    images=curr_images,
                )

                # Make prediction.
                pred, status, response = models[model].forward(
                    messages=messages, client_kwargs=client_kwargs
                )

                results[-1][f"{exp_key}_pred"] = pred
                results[-1][f"{exp_key}_status"] = status
                results[-1][f"{exp_key}_response"] = response

                # Debug print responses.
                dbprint(
                    f"Model: {model} | Template: {template} | Episode: {data[-1]['episode']} | Failure: {not data[-1]['success']} | Timestep: {data[-1]['timestep']} | Response:"
                )
                dbprint(response)
                if BREAKPOINT:
                    breakpoint()

    results_frame = pd.DataFrame(results)
    return results_frame


def evaluate_vlm(
    cfg: Dict,
    ref_dataset_path: Union[str, pathlib.Path],
    test_dataset_path: Union[str, pathlib.Path],
) -> Dict[str, Union[Dict[str, Any], pd.DataFrame]]:
    """Compute vision-language model results."""
    # Construct episode iterable datasets.
    ref_dataset = VideoDataset(
        dataset_path=ref_dataset_path,
        exec_horizon=1 if cfg.eval.calib_on_light else cfg.model.ac_horizon,
        num_timesteps=getattr(cfg.eval.vlm, "num_timesteps"),
        subsample_freq=getattr(cfg.eval.vlm, "subsample_freq"),
        filter_success=getattr(cfg.eval, "filter_demo_success", False),
        filter_failure=getattr(cfg.eval, "filter_demo_failure", True),
        filter_episodes=getattr(cfg.eval, "filter_demo_episodes", None),
        max_episode_length=getattr(cfg.eval.vlm, "max_video_length", None),
        max_num_episodes=getattr(cfg.eval, "max_num_episodes", None),
    )
    test_dataset = VideoDataset(
        dataset_path=test_dataset_path,
        exec_horizon=1,
        num_timesteps=getattr(cfg.eval.vlm, "num_timesteps"),
        subsample_freq=getattr(cfg.eval.vlm, "subsample_freq"),
        filter_success=getattr(cfg.eval, "filter_test_success", False),
        filter_failure=getattr(cfg.eval, "filter_test_failure", False),
        filter_episodes=getattr(cfg.eval, "filter_test_episodes", None),
        max_episode_length=getattr(cfg.eval.vlm, "max_video_length", None),
        max_num_episodes=getattr(cfg.eval, "max_num_episodes", None),
    )

    # Compute predictions for specified parameter sets.
    results_dict = {}
    test_results_frame = compute_vlm_predictions(cfg, ref_dataset, test_dataset)

    # Compute metrics for specified parameter sets.
    for model in cfg.eval.vlm.models:
        for template in getattr(cfg.eval.vlm.templates, model):
            exp_key = vlm_exp_key(model=model, template=template)
            metric_utils.compute_prediction_results(
                exp_key=exp_key,
                results_dict=results_dict,
                test_results_frame=test_results_frame,
            )

    return {
        "results_dict": results_dict,
        "test_results_frame": test_results_frame,
    }


def compute_temporal_consistency_errors(
    cfg: omegaconf.DictConfig, dataset: EpisodeDataset
) -> pd.DataFrame:
    """Compute temporal consistency errors over dataset."""
    results = []
    exp_keys = []

    for prev_data, curr_data in iter(dataset):
        assert curr_data["timestep"] - prev_data["timestep"] == cfg.model.ac_horizon

        results.append(
            {
                "episode": curr_data["episode"],
                "timestep": curr_data["timestep"],
                "success": curr_data.get("success", True),
            }
        )
        rgb = get_rgb(curr_data)
        if isinstance(rgb, np.ndarray):
            results[-1]["rgb"] = rgb

        for sample_size in cfg.eval.consistency.sample_sizes:

            # Subsample current and previous actions.
            curr_actions = action_utils.subsample_actions(
                curr_data["sampled_actions"],
                sample_size,
            )
            curr_skip_steps = curr_data.get("skip_steps", None)

            prev_actions = action_utils.subsample_actions(
                prev_data["sampled_actions"],
                sample_size,
            )
            prev_skip_steps = prev_data.get("skip_steps", None)

            for error_fn in cfg.eval.consistency.error_fns:

                if error_fn in CONSISTENCY_ERROR_FNS:
                    prev_selected_actions = prev_data["executed_action"]
                    error_fn_kwargs = CONSISTENCY_ERROR_FNS[error_fn]
                elif error_fn in CONSISTENCY_DIST_ERROR_FNS:
                    prev_selected_actions = prev_actions
                    error_fn_kwargs = CONSISTENCY_DIST_ERROR_FNS[error_fn]
                else:
                    raise ValueError(f"Error function {error_fn} not supported.")

                for aggr_fn in get_consistency_aggr_fns(cfg, error_fn):
                    for pred_horizon in cfg.eval.consistency.pred_horizons:
                        if cfg.model.ac_horizon >= pred_horizon:
                            continue

                        exp_key = temporal_consistency_exp_key(
                            pred_horizon=pred_horizon,
                            sample_size=sample_size,
                            error_fn=error_fn,
                            aggr_fn=aggr_fn,
                        )
                        if exp_key not in exp_keys:
                            exp_keys.append(exp_key)

                        error = error_utils.compute_temporal_error(
                            curr_action=curr_actions,
                            prev_action=prev_selected_actions,
                            pred_horizon=pred_horizon,
                            exec_horizon=cfg.model.ac_horizon,
                            sim_freq=cfg.env.args.freq,
                            num_robots=cfg.env.num_eef,
                            action_dim=cfg.env.dof,
                            skip_steps=getattr(
                                cfg.eval.consistency, "skip_steps", False
                            ),
                            curr_skip_steps=curr_skip_steps,
                            prev_skip_steps=prev_skip_steps,
                            **error_fn_kwargs,
                        )
                        if error_fn in CONSISTENCY_ERROR_FNS:
                            error = CONSISTENCY_AGGR_FNS[aggr_fn](error)
                        results[-1][f"{exp_key}_score"] = error

    results_frame = compute_cum_scores(pd.DataFrame(results), exp_keys)
    return results_frame


def evaluate_temporal_consistency(
    cfg: omegaconf.DictConfig,
    demo_dataset_path: Union[str, pathlib.Path],
    test_dataset_path: Union[str, pathlib.Path],
) -> Dict[str, Union[Dict[str, Any], pd.DataFrame]]:
    """Compute temporal consistency results."""
    # Construct episode iterable datasets.
    demo_dataset = EpisodeDataset(
        dataset_path=demo_dataset_path,
        exec_horizon=1 if cfg.eval.calib_on_light else cfg.model.ac_horizon,
        sample_history=1,
        filter_success=getattr(cfg.eval, "filter_demo_success", False),
        filter_failure=getattr(cfg.eval, "filter_demo_failure", True),
        filter_episodes=getattr(cfg.eval, "filter_demo_episodes", None),
        max_episode_length=getattr(cfg.eval, "max_demo_episode_length", None),
        max_num_episodes=getattr(cfg.eval, "max_num_episodes", None),
    )
    test_dataset = EpisodeDataset(
        dataset_path=test_dataset_path,
        exec_horizon=1,
        sample_history=1,
        filter_success=getattr(cfg.eval, "filter_test_success", False),
        filter_failure=getattr(cfg.eval, "filter_test_failure", False),
        filter_episodes=getattr(cfg.eval, "filter_test_episodes", None),
        max_episode_length=getattr(cfg.eval, "max_test_episode_length", None),
        max_num_episodes=getattr(cfg.eval, "max_num_episodes", None),
    )

    # Compute scores for specified parameter sets.
    results_dict = defaultdict(dict)
    demo_results_frame = compute_temporal_consistency_errors(cfg, demo_dataset)
    test_results_frame = compute_temporal_consistency_errors(cfg, test_dataset)

    # Compute metrics for specified parameter sets.
    for sample_size in cfg.eval.consistency.sample_sizes:
        for error_fn in cfg.eval.consistency.error_fns:
            for aggr_fn in get_consistency_aggr_fns(cfg, error_fn):
                for pred_horizon in cfg.eval.consistency.pred_horizons:
                    if cfg.model.ac_horizon >= pred_horizon:
                        continue

                    exp_key = temporal_consistency_exp_key(
                        pred_horizon=pred_horizon,
                        sample_size=sample_size,
                        error_fn=error_fn,
                        aggr_fn=aggr_fn,
                    )

                    for quantile in cfg.eval.quantiles:

                        test_results_frame = metric_utils.compute_detection_results(
                            exp_key=exp_key,
                            quantile_key=quantile_exp_key(exp_key, quantile),
                            results_dict=results_dict,
                            demo_results_frame=demo_results_frame,
                            test_results_frame=test_results_frame,
                            detector=getattr(cfg.eval, "detector", "quantile"),
                            detector_kwargs={
                                "quantile": quantile,
                                **getattr(cfg.eval, "detector_kwargs", {}),
                            },
                        )

    return {
        "results_dict": results_dict,
        "test_results_frame": test_results_frame,
        "demo_results_frame": demo_results_frame,
    }


@hydra.main(config_path="configs", config_name="close_mobile_dp")
def main(cfg: omegaconf.DictConfig):
    assert cfg.mode == "eval"
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Configure paths.
    log_dir = pathlib.Path(os.getcwd())
    demo_dataset_path = log_dir / ".." / cfg.eval.demo_experiment / "episodes"
    test_dataset_path = log_dir / "episodes"
    output_path = log_dir / str(cfg.eval.output_path)
    output_path.mkdir(exist_ok=True)

    # Temporal consistency evaluation.
    if getattr(cfg.eval, "evaluate_temporal_consistency", False):
        print("\n\nEvaluating Temporal Consistency")
        temporal_consistency_results = evaluate_temporal_consistency(
            cfg,
            demo_dataset_path=demo_dataset_path,
            test_dataset_path=test_dataset_path,
        )
        with open(output_path / "temporal_consistency_results.pkl", "wb") as f:
            pickle.dump(
                temporal_consistency_results, f, protocol=pickle.HIGHEST_PROTOCOL
            )
        del temporal_consistency_results

    # Vision-language model approaches.
    if getattr(cfg.eval, "evaluate_vlm", False):
        print("\n\nEvaluating Vision-Language Models")
        ref_experiment = getattr(cfg.eval.vlm, "ref_experiment", None)
        assert ref_experiment is not None, "Must specify a reference experiment."
        ref_dataset_path = log_dir / ".." / ref_experiment / "episodes"
        vlm_results = evaluate_vlm(
            cfg,
            ref_dataset_path=ref_dataset_path,
            test_dataset_path=test_dataset_path,
        )
        with open(output_path / f"vlm_results.pkl", "wb") as f:
            pickle.dump(vlm_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        del vlm_results

    # Diffusion ensemble evaluation.
    if getattr(cfg.eval, "evaluate_diffusion_ensemble", False):
        print("\n\nEvaluating Diffusion Ensemble")
        diffusion_ensemble_results = evaluate_diffusion_ensemble(
            cfg,
            demo_dataset_path=demo_dataset_path,
            test_dataset_path=test_dataset_path,
        )
        with open(output_path / f"diffusion_ensemble_results.pkl", "wb") as f:
            pickle.dump(diffusion_ensemble_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        del diffusion_ensemble_results

    # Embedding similarity evaluation.
    if getattr(cfg.eval, "evaluate_embedding_similarity", False):
        print("\n\nEvaluating Embedding Similarity")
        embedding_similarity_results = evaluate_embedding_similarity(
            cfg,
            demo_dataset_path=demo_dataset_path,
            test_dataset_path=test_dataset_path,
        )
        with open(output_path / f"embedding_similarity_results.pkl", "wb") as f:
            pickle.dump(
                embedding_similarity_results, f, protocol=pickle.HIGHEST_PROTOCOL
            )
        del embedding_similarity_results

    # Loss function approaches.
    if getattr(cfg.eval, "evaluate_loss_functions", False):
        print("\n\nEvaluating Loss Functions")
        loss_function_results = evaluate_loss_functions(
            cfg,
            demo_dataset_path=demo_dataset_path,
            test_dataset_path=test_dataset_path,
        )
        with open(output_path / f"loss_function_results.pkl", "wb") as f:
            pickle.dump(loss_function_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        del loss_function_results


if __name__ == "__main__":
    main()

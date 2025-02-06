from typing import Union, Dict, Any, List, Tuple, Optional

import os
import sys
import pickle
import pathlib
import numpy as np
import pandas as pd

root_dir = pathlib.Path(os.getcwd()).parent
sys.path.append(str(root_dir / "sentinel" / "point_feat"))

from sentinel.bc.evaluate_ood_detectors import (
    temporal_consistency_exp_key,
    vlm_exp_key,
    diffusion_ensemble_exp_key,
    embedding_similarity_exp_key,
    loss_function_exp_key,
    quantile_exp_key,
)
import sentinel.bc.datasets.utils as data_utils
from sentinel.bc.ood_detection import metric_utils


CWD = pathlib.Path(os.getcwd())
LOG_DIR = pathlib.Path("../../logs/bc/eval")
REAL_LOG_DIR = pathlib.Path("../../logs/bc/real_eval")
EXP_DIR = {
    #### Archived datasets. ####
    # Close hyperparameter sweep datasets.
    "0525_close_4_sweep_na": "0525_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    "0525_close_4_sweep_ll": "0525_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_ll_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    "0525_close_4_sweep_hh": "0525_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    # Close, Cover visualization datasets.
    "0525_close_4_abl_ss": "0525_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_ss_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    "0527_cover_4_abl_ss": "0527_sim_compute_rollout_actions_cover_dp_ckpt00999_exec_horizon_4_randomization_ss_rigid_pos_1_rigid_scale_0_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    #### Official result datasets. ####
    # PushT main result (Figure 5): All methods sans VLM.
    "0525_pusht_8_na": "0525_sim_compute_rollout_actions_pusht_dp_aug_ckpt01999_exec_horizon_8_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_0_dynamics_noise_0",
    "0525_pusht_8_hh": "0525_sim_compute_rollout_actions_pusht_dp_aug_ckpt01999_exec_horizon_8_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_0_dynamics_noise_0",
    "0526_pusht_8_na": "0526_sim_compute_rollout_actions_pusht_dp_aug_ckpt01999_exec_horizon_8_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_0_dynamics_noise_0",
    "0526_pusht_8_hh": "0526_sim_compute_rollout_actions_pusht_dp_aug_ckpt01999_exec_horizon_8_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_0_dynamics_noise_0",
    "0527_pusht_8_na": "0527_sim_compute_rollout_actions_pusht_dp_aug_ckpt01999_exec_horizon_8_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_0_dynamics_noise_0",
    "0527_pusht_8_hh": "0527_sim_compute_rollout_actions_pusht_dp_aug_ckpt01999_exec_horizon_8_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_0_dynamics_noise_0",
    # PushT ablation result (Figure 8, Figure 9): STAC.
    "0525_pusht_2_na": "0525_sim_compute_rollout_actions_pusht_dp_aug_ckpt01999_exec_horizon_2_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_0_dynamics_noise_0",
    "0525_pusht_2_hh": "0525_sim_compute_rollout_actions_pusht_dp_aug_ckpt01999_exec_horizon_2_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_0_dynamics_noise_0",
    "0525_pusht_4_na": "0525_sim_compute_rollout_actions_pusht_dp_aug_ckpt01999_exec_horizon_4_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_0_dynamics_noise_0",
    "0525_pusht_4_hh": "0525_sim_compute_rollout_actions_pusht_dp_aug_ckpt01999_exec_horizon_4_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_0_dynamics_noise_0",
    # Close erratic failures (Table 1, Table 5): All methods sans VLM.
    "0527_close_4_na": "0527_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    "0527_close_4_hh": "0527_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    "0528_close_4_na": "0528_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    "0528_close_4_hh": "0528_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    "0529_close_4_na": "0529_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    "0529_close_4_hh": "0529_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    # Close erratic failures (Table 1, Table 5): STAC and VLM (Sentinel).
    "0914_close_4_na": "0826_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    "0914_close_4_hh": "0826_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_hh_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    # Close task progression failures (Figure 6, Table 7): STAC and VLM (Sentinel).
    "0914_close_4_ss": "0826_sim_compute_rollout_actions_close_dp_ckpt00999_exec_horizon_4_randomization_ss_rigid_pos_0_rigid_scale_1_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    # Cover task progression failures (Figure 6, Table 6): STAC and VLM (Sentinel).
    "0914_cover_4_na": "0826_sim_compute_rollout_actions_cover_dp_ckpt00999_exec_horizon_4_randomization_na_rigid_pos_1_rigid_scale_0_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
    "0914_cover_4_ss": "0904_sim_compute_rollout_actions_cover_dp_ckpt00999_exec_horizon_4_randomization_ss_rigid_pos_1_rigid_scale_0_soft_dynamics_0_robot_eef_pos_1_dynamics_noise_0",
}
REAL_EXP_DIR = {
    # Push chair real-world result (Table 2): STAC and VLM (Sentinel).
    "0914_push_chair_4_test": "0914_push_chair_sim3_dp_ddim_test",
}

RES_DIR = {
    #### Archived datasets. ####
    # Close hyperparameter sweep datasets.
    "0525_close_4_sweep_na": "0527_sweep_results_calib_on_light_1",
    "0525_close_4_sweep_ll": "0527_sweep_results_calib_on_light_1",
    "0525_close_4_sweep_hh": "0527_sweep_results_calib_on_light_1",
    # Close, Cover visualization datasets.
    "0525_close_4_abl_ss": "0612_results_calib_on_light_1",
    "0527_cover_4_abl_ss": "0612_results_calib_on_light_1",
    #### Official result datasets. ####
    # PushT main result (Figure 5): All methods sans VLM.
    "0525_pusht_8_na": "0526_results_calib_on_light_1",
    "0525_pusht_8_hh": "0526_results_calib_on_light_1",
    "0526_pusht_8_na": "0528_results_calib_on_light_1",
    "0526_pusht_8_hh": "0528_results_calib_on_light_1",
    "0527_pusht_8_na": "0528_results_calib_on_light_1",
    "0527_pusht_8_hh": "0528_results_calib_on_light_1",
    # PushT ablation result (Figure 8, Figure 9): STAC.
    "0525_pusht_2_na": "0526_results_calib_on_light_1",
    "0525_pusht_2_hh": "0526_results_calib_on_light_1",
    "0525_pusht_4_na": "0526_results_calib_on_light_1",
    "0525_pusht_4_hh": "0526_results_calib_on_light_1",
    # Close erratic failures (Table 1, Table 5): All methods sans VLM.
    "0527_close_4_na": "0528_results_calib_on_light_1",
    "0527_close_4_hh": "0528_results_calib_on_light_1",
    "0528_close_4_na": "0528_results_calib_on_light_1",
    "0528_close_4_hh": "0528_results_calib_on_light_1",
    "0529_close_4_na": "0528_results_calib_on_light_1",
    "0529_close_4_hh": "0528_results_calib_on_light_1",
    # Close erratic failures (Table 1, Table 5): STAC and VLM (Sentinel).
    "0914_close_4_na": "0914_results_calib_on_light_1",
    "0914_close_4_hh": "0914_results_calib_on_light_1",
    # Close task progression failures (Figure 6, Table 7): STAC and VLM (Sentinel).
    "0914_close_4_ss": "0914_results_calib_on_light_1",
    # Cover task progression failures (Figure 6, Table 6): STAC and VLM (Sentinel).
    "0914_cover_4_na": "0914_results_calib_on_light_1",
    "0914_cover_4_ss": "0914_results_calib_on_light_1",
    # Push chair real-world result (Table 2): STAC and VLM (Sentinel).
    "0914_push_chair_4_test": "0914_results_calib_on_light_1",
}


# Results utils.
quantile = 0.95


def load_pickle(path: Union[str, pathlib.Path]) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        data = None
    return data


def load_result(
    domain: str,
    split: str,
) -> Dict[str, Dict[str, Any]]:
    result_files = [
        "loss_function_results.pkl",
        "temporal_consistency_results.pkl",
        "diffusion_ensemble_results.pkl",
        "embedding_similarity_results.pkl",
        "vlm_results.pkl",
    ]
    k = f"{domain}_{split}"
    if k in EXP_DIR:
        result_path = LOG_DIR / EXP_DIR[k] / RES_DIR[k]
    elif k in REAL_EXP_DIR:
        result_path = REAL_LOG_DIR / REAL_EXP_DIR[k] / RES_DIR[k]
    return {f: load_pickle(result_path / f) for f in result_files}


def load_episode_frame(
    domain: str,
    split: str,
    episode: int,
) -> pd.DataFrame:
    """Load episode rollout DataFrame with saved tensors."""
    k = f"{domain}_{split}"
    if k in EXP_DIR:
        episode_path = LOG_DIR / EXP_DIR[k]
    elif k in REAL_EXP_DIR:
        episode_path = REAL_LOG_DIR / REAL_EXP_DIR[k]
    episode_path = episode_path / "episodes" / f"ep{episode:04d}.pkl"
    return load_pickle(episode_path)


def get_loss_function_exp_keys(
    loss_fns: List[str], sample_sizes: List[int]
) -> List[str]:
    exp_keys = []
    for l in loss_fns:
        for s in sample_sizes:
            exp_key = loss_function_exp_key(loss_fn=l, sample_size=s)
            if exp_key not in exp_keys:
                exp_keys.append(exp_key)
    return exp_keys


def get_temporal_consistency_exp_keys(
    pred_horizons: List[int],
    sample_sizes: List[int],
    error_fns: List[str],
    aggr_fns: List[str],
) -> List[str]:
    exp_keys = []
    for h in pred_horizons:
        for s in sample_sizes:
            for e in error_fns:
                for a in aggr_fns:
                    exp_key = temporal_consistency_exp_key(
                        pred_horizon=h,
                        sample_size=s,
                        error_fn=e,
                        aggr_fn=a,
                    )
                    if exp_key not in exp_keys:
                        exp_keys.append(exp_key)
    return exp_keys


def get_vlm_exp_keys(
    models: List[str],
    templates: Dict[str, List[str]],
) -> List[str]:
    exp_keys = []
    for m in models:
        for t in templates[m]:
            exp_key = vlm_exp_key(m, t)
            if exp_key not in exp_keys:
                exp_keys.append(exp_key)
    return exp_keys


def get_ensemble_exp_keys(
    pred_horizons: List[int],
    sample_sizes: List[int],
    action_spaces: List[str],
) -> List[str]:
    exp_keys = []
    for h in pred_horizons:
        for s in sample_sizes:
            for a in action_spaces:
                exp_key = diffusion_ensemble_exp_key(
                    pred_horizon=h,
                    sample_size=s,
                    action_space=a,
                )
                if exp_key not in exp_keys:
                    exp_keys.append(exp_key)
    return exp_keys


def get_embedding_exp_keys(
    embeddings: List[str],
    score_fns: List[str],
) -> List[str]:
    exp_keys = []
    for e in embeddings:
        for s in score_fns:
            exp_key = embedding_similarity_exp_key(
                embedding=e,
                score_fn=s,
            )
            if exp_key not in exp_keys:
                exp_keys.append(exp_key)
    return exp_keys


def exp_key_to_result_file(exp_key: str) -> str:
    if "loss_fn" in exp_key:
        result_file = "loss_function_results.pkl"
    elif "error_fn" in exp_key:
        result_file = "temporal_consistency_results.pkl"
    elif "action_space" in exp_key:
        result_file = "diffusion_ensemble_results.pkl"
    elif "embedding" in exp_key:
        result_file = "embedding_similarity_results.pkl"
    elif "model" in exp_key:
        result_file = "vlm_results.pkl"
    else:
        raise ValueError(f"Experiment key {exp_key} not recognized.")
    return result_file


def compile_metrics(
    domain: str,
    splits: List[str],
    exp_keys: List[str],
    quantile: Optional[float] = 0.95,
    calib_key: Optional[str] = "ep_iid_cum",
    return_test_data: bool = False,
    return_test_frame: bool = False,
    return_demo_frame: bool = False,
) -> Dict[str, Any]:
    metrics = {}
    for split in splits:
        metrics[split] = {}

        # Load results.
        results_dict = load_result(
            domain=domain,
            split=split,
        )

        # Store results.
        for exp_key in exp_keys:
            results_file = exp_key_to_result_file(exp_key)
            is_vlm = "vlm" in results_file
            quantile_key = exp_key if is_vlm else quantile_exp_key(exp_key, quantile)

            # Check if result exists.
            if (
                results_file in results_dict
                and quantile_key in results_dict[results_file]["results_dict"]
            ):
                metrics[split][exp_key] = {}

                # VLM results do not use calibration.
                detection_results = results_dict[results_file]["results_dict"][
                    quantile_key
                ]
                if not is_vlm:
                    detection_results = detection_results[calib_key]
                metrics[split][exp_key]["metrics"] = detection_results["episode"][
                    "metrics"
                ]
                metrics[split][exp_key]["data"] = detection_results["episode"]["data"]

                if return_test_data:
                    metrics[split][exp_key]["test_data"] = detection_results["episode"][
                        "data"
                    ]
                if return_test_frame:
                    metrics[split][exp_key]["test_frame"] = results_dict[results_file][
                        "test_results_frame"
                    ]
                if return_demo_frame:
                    metrics[split][exp_key]["demo_frame"] = results_dict[results_file][
                        "demo_results_frame"
                    ]

    return metrics


def aggregate_metrics(
    splits: List[str],
    exp_keys: List[str],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    aggr_metrics = {}
    for exp_key in exp_keys:
        if all(exp_key in data[s] for s in splits):
            aggr_metrics[exp_key] = {}
            TP = TN = FP = FN = 0
            for split in splits:
                TP += data[split][exp_key]["metrics"]["TP"]
                TN += data[split][exp_key]["metrics"]["TN"]
                FP += data[split][exp_key]["metrics"]["FP"]
                FN += data[split][exp_key]["metrics"]["FN"]
            aggr_metrics[exp_key]["metrics"] = metric_utils.compute_metrics(
                TP=TP, TN=TN, FP=FP, FN=FN
            )
    return aggr_metrics


def extract_metric_list(
    exp_keys: List[str],
    data: Dict[str, Any],
    metric: str = "Accuracy",
) -> np.ndarray:
    metrics = np.array(
        [
            float(data[exp_key]["metrics"][metric])
            for exp_key in exp_keys
            if exp_key in data
        ]
    )
    return metrics


def extract_metric_dict(
    exp_keys: List[str],
    data: Dict[str, Any],
    metric: str = "Accuracy",
) -> Dict[str, float]:
    return {
        exp_key: data[exp_key]["metrics"][metric]
        for exp_key in exp_keys
        if exp_key in data
    }


def sort_metrics(
    exp_keys: List[str],
    data: Dict[str, Any],
    metric: str = "Accuracy",
    reverse: bool = True,
) -> Tuple[List[str], List[Any]]:
    scores = extract_metric_list(exp_keys, data, metric)
    indices = scores.argsort()

    scores = scores[indices].tolist()
    exp_keys = np.array(exp_keys)[indices].tolist()

    if reverse:
        scores = scores[::-1]
        exp_keys = exp_keys[::-1]

    return exp_keys, scores


def get_detection_data(
    split: str,
    exp_key: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    episode_data: Dict[str, Any] = data[split][exp_key]["test_data"]
    test_scores = episode_data.get("test_scores", None)
    test_preds = episode_data["test_preds"]
    test_labels = ~episode_data["test_labels"]
    test_times = episode_data["test_detection_times"]

    TP = np.sum((test_preds == True) & (test_labels == True))
    TN = np.sum((test_preds == False) & (test_labels == False))
    FP = np.sum((test_preds == True) & (test_labels == False))
    FN = np.sum((test_preds == False) & (test_labels == True))

    detection_data = {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "P_idx": np.where(test_labels == True)[0],
        "N_idx": np.where(test_labels == False)[0],
        "TP_idx": np.where((test_preds == True) & ((test_labels) == True))[0],
        "TN_idx": np.where((test_preds == False) & ((test_labels) == False))[0],
        "FP_idx": np.where((test_preds == True) & ((test_labels) == False))[0],
        "FN_idx": np.where((test_preds == False) & ((test_labels) == True))[0],
        "test_scores": test_scores,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "test_times": test_times,
    }

    return detection_data


def aggr_detection_data(
    data: List[Dict[str, Any]],
    use_preds: bool = False,
    use_times: bool = False,
) -> Dict[str, Any]:

    if use_preds:
        test_preds = np.concatenate([d["test_preds"] for d in data])
        test_labels = np.concatenate([d["test_labels"] for d in data])
        test_times = (
            np.concatenate([d["test_times"] for d in data]) if use_times else None
        )
        metrics = metric_utils.compute_metrics(
            preds=test_preds,
            labels=test_labels,
            times=test_times,
        )
    else:
        TP = TN = FP = FN = 0
        for d in data:
            TP += d["TP"]
            TN += d["TN"]
            FP += d["FP"]
            FN += d["FN"]
        metrics = metric_utils.compute_metrics(TP=TP, TN=TN, FP=FP, FN=FN)

    return metrics


def get_detection_preds(
    split: str,
    exp_key: str,
    data: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Retrieve success and failure scores."""
    test_frame = data[split][exp_key]["test_frame"]
    test_detection_data = get_detection_data(split, exp_key, data)

    num_episodes = data_utils.num_episodes(test_frame)
    times = data_utils.get_episode(test_frame, 0, use_index=True)["timestep"].values

    preds = np.zeros((num_episodes, len(times)))
    for i in range(data_utils.num_episodes(test_frame)):
        episode_frame = data_utils.get_episode(test_frame, i, use_index=True)
        preds[i, :] = episode_frame[f"{exp_key}_pred"].values

    return {
        "preds": preds,
        "timesteps": times,
        "P_preds": preds[test_detection_data["P_idx"]],
        "N_preds": preds[test_detection_data["N_idx"]],
        "TP_preds": preds[test_detection_data["TP_idx"]],
        "TN_preds": preds[test_detection_data["TN_idx"]],
        "FP_preds": preds[test_detection_data["FP_idx"]],
        "FN_preds": preds[test_detection_data["FN_idx"]],
    }


def get_detection_scores(
    split: str,
    exp_key: str,
    data: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Retrieve success and failure scores."""
    test_frame = data[split][exp_key]["test_frame"]
    test_detection_data = get_detection_data(split, exp_key, data)

    num_episodes = data_utils.num_episodes(test_frame)
    times = data_utils.get_episode(test_frame, 0, use_index=True)["timestep"].values

    scores = np.zeros((num_episodes, len(times)))
    for i in range(data_utils.num_episodes(test_frame)):
        episode_frame = data_utils.get_episode(test_frame, i, use_index=True)
        scores[i, :] = episode_frame[f"{exp_key}_cum_score"].values

    return {
        "scores": scores,
        "timesteps": times,
        "P_scores": scores[test_detection_data["P_idx"]],
        "N_scores": scores[test_detection_data["N_idx"]],
        "TP_scores": scores[test_detection_data["TP_idx"]],
        "TN_scores": scores[test_detection_data["TN_idx"]],
        "FP_scores": scores[test_detection_data["FP_idx"]],
        "FN_scores": scores[test_detection_data["FN_idx"]],
    }


def ensemble_vlm_detection_data(
    split: str,
    exp_keys: List[str],
    data: Dict[str, Any],
    strategy: Optional[str] = "majority_vote",
    thresh: Optional[int] = None,
) -> Dict[str, Any]:
    assert (strategy is not None) ^ (thresh is not None)

    # Ensemble predictions per timestep: Majority vote.
    if strategy == "majority_vote":
        thresh = len(exp_keys) // 2

    # Compute ensemble prediction.
    detection_data = [get_detection_preds(split, exp_key, data) for exp_key in exp_keys]
    vlm_preds: np.ndarray = np.stack(
        [d["preds"] for d in detection_data], axis=-1
    )  # [Num. Episodes, Timesteps, Ensemble]
    ens_preds: np.ndarray = (
        np.sum(vlm_preds, axis=-1) > thresh
    )  # [Num. Episodes, Timesteps]
    test_preds: np.ndarray = np.any(ens_preds, axis=-1)  # [Num. Episodes]

    # Compute detection times.
    num_episodes = len(test_preds)
    timesteps = detection_data[0]["timesteps"]
    test_times = np.zeros(num_episodes, dtype=float)
    for i in range(num_episodes):
        if test_preds[i]:
            pred_idx: int = np.where(ens_preds[i] == True)[0][0]
        else:
            pred_idx = -1
        test_times[i] = timesteps[pred_idx]

    # Compute detection data.
    test_labels: np.ndarray = get_detection_data(split, exp_keys[0], data)[
        "test_labels"
    ]
    assert test_labels.shape == test_preds.shape

    TP = np.sum((test_preds == True) & (test_labels == True))
    TN = np.sum((test_preds == False) & (test_labels == False))
    FP = np.sum((test_preds == True) & (test_labels == False))
    FN = np.sum((test_preds == False) & (test_labels == True))

    detection_data = {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "P_idx": np.where(test_labels == True)[0],
        "N_idx": np.where(test_labels == False)[0],
        "TP_idx": np.where((test_preds == True) & ((test_labels) == True))[0],
        "TN_idx": np.where((test_preds == False) & ((test_labels) == False))[0],
        "FP_idx": np.where((test_preds == True) & ((test_labels) == False))[0],
        "FN_idx": np.where((test_preds == False) & ((test_labels) == True))[0],
        "test_scores": None,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "test_times": test_times,
    }

    return detection_data


def ensemble_detection_data(
    *detection_data: Dict[str, Any],
) -> Dict[str, Any]:
    # Ensure all episode labels are identical.
    test_labels: np.ndarray = detection_data[0]["test_labels"]
    assert all(np.all(test_labels == d["test_labels"]) for d in detection_data)

    test_preds = np.zeros_like(test_labels, dtype=bool)
    test_times = np.ones_like(test_labels, dtype=float) * np.inf
    for d in detection_data:
        test_preds: np.ndarray = np.logical_or(test_preds, d["test_preds"])
        test_times: np.ndarray = np.minimum(test_times, d["test_times"])

    TP = np.sum((test_preds == True) & (test_labels == True))
    TN = np.sum((test_preds == False) & (test_labels == False))
    FP = np.sum((test_preds == True) & (test_labels == False))
    FN = np.sum((test_preds == False) & (test_labels == True))

    detection_data = {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "P_idx": np.where(test_labels == True)[0],
        "N_idx": np.where(test_labels == False)[0],
        "TP_idx": np.where((test_preds == True) & ((test_labels) == True))[0],
        "TN_idx": np.where((test_preds == False) & ((test_labels) == False))[0],
        "FP_idx": np.where((test_preds == True) & ((test_labels) == False))[0],
        "FN_idx": np.where((test_preds == False) & ((test_labels) == True))[0],
        "test_scores": None,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "test_times": test_times,
    }

    return detection_data


def print_metrics(
    prefix: str,
    metrics: Dict[str, Any],
    with_accuracy: bool = False,
    time_mod: Optional[float] = None,
) -> None:
    t = f"{prefix} | "
    if with_accuracy:
        keys = ["TPR", "TNR", "FPR", "Accuracy", "TP Time Mean"]
    else:
        keys = ["TPR", "TNR", "FPR", "TP Time Mean"]
    for k in keys:
        if k in metrics:
            if k == "TP Time Mean" and time_mod is not None:
                t += f"{k}: {metrics[k] / time_mod:.2f} | "
            else:
                t += f"{k}: {metrics[k]:.2f} | "
    print(t.strip())


def compute_sentinel_result(
    stac_exp_key: str,
    vlm_exp_keys_list: List[List[str]],
    splits_list: List[List[str]],
    metrics_list: List[Dict[str, Any]],
    time_mod: Optional[float] = None,
    domain_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    stac_detection_data = []
    vlme_detection_data = []
    sent_detection_data = []
    for i, (domain_metrics, domain_splits, domain_exp_keys) in enumerate(
        zip(metrics_list, splits_list, vlm_exp_keys_list)
    ):
        if domain_names is not None:
            print(f"\nDomain: {domain_names[i]}")
        else:
            print(f"\nDomain: {i}")

        for split in domain_splits:
            print(f"\nSplit: {split}")

            # STAC detection data.
            stac_data = get_detection_data(split, stac_exp_key, domain_metrics)
            stac_detection_data.append(stac_data)
            stac_metrics = metric_utils.compute_metrics(
                preds=stac_data["test_preds"],
                labels=stac_data["test_labels"],
                times=stac_data["test_times"],
            )
            print_metrics(
                f"STAC: {split}", stac_metrics, with_accuracy=True, time_mod=time_mod
            )

            # VLM detection data.
            vlme_data = ensemble_vlm_detection_data(
                split, domain_exp_keys, domain_metrics
            )
            vlme_detection_data.append(vlme_data)
            vlme_metrics = metric_utils.compute_metrics(
                preds=vlme_data["test_preds"],
                labels=vlme_data["test_labels"],
                times=vlme_data["test_times"],
            )
            print_metrics(
                f"VLME: {split}", vlme_metrics, with_accuracy=True, time_mod=time_mod
            )

            # Sentinel detection data.
            sent_data = ensemble_detection_data(stac_data, vlme_data)
            sent_detection_data.append(sent_data)
            sentinel_metrics = metric_utils.compute_metrics(
                preds=sent_data["test_preds"],
                labels=sent_data["test_labels"],
                times=sent_data["test_times"],
            )
            print_metrics(
                f"SENT: {split}",
                sentinel_metrics,
                with_accuracy=True,
                time_mod=time_mod,
            )

    print("\nAggregate Results")
    stac_metrics = aggr_detection_data(
        stac_detection_data, use_preds=True, use_times=True
    )
    print_metrics("STAC", stac_metrics, with_accuracy=True, time_mod=time_mod)
    vlme_metrics = aggr_detection_data(
        vlme_detection_data, use_preds=True, use_times=True
    )
    print_metrics("VLME", vlme_metrics, with_accuracy=True, time_mod=time_mod)
    sent_metrics = aggr_detection_data(
        sent_detection_data, use_preds=True, use_times=True
    )
    print_metrics("SENT", sent_metrics, with_accuracy=True, time_mod=time_mod)

    return stac_metrics, vlme_metrics, sent_metrics

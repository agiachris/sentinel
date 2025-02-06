from typing import Optional, Any, Dict

import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics

import sentinel.bc.datasets.utils as data_utils
from sentinel.bc.ood_detection import get_detector


DEBUG = 1
dbprint = print if DEBUG == 1 else lambda *args: ()


def compute_metrics(
    preds: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    times: Optional[np.ndarray] = None,
    success_labels: bool = False,
    TP: Optional[int] = None,
    TN: Optional[int] = None,
    FP: Optional[int] = None,
    FN: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute detection metrics."""
    metrics = {}

    if preds is not None and labels is not None:
        if success_labels:
            labels = ~labels
        TP = np.sum((preds == 1) & (labels == 1))
        TN = np.sum((preds == 0) & (labels == 0))
        FP = np.sum((preds == 1) & (labels == 0))
        FN = np.sum((preds == 0) & (labels == 1))

        # Detection times.
        if times is not None and TP > 0:
            tp_times = times[np.where((preds == 1) & (labels == 1))[0]]
            metrics["TP Time Mean"] = tp_times.mean()
            metrics["TP Time STD"] = tp_times.std()
        if times is not None and FP > 0:
            fp_times = times[np.where((preds == 1) & (labels == 0))[0]]
            metrics["FP Time Mean"] = fp_times.mean()
            metrics["FP Time STD"] = fp_times.std()

        # AUROC score.
        if scores is not None and (labels == 0).sum() > 0 and (labels == 1).sum() > 0:
            metrics["AUROC Score"] = sklearn_metrics.roc_auc_score(labels, scores)

    # Store TP, TN, FP, FN.
    metrics["TP"] = TP
    metrics["TN"] = TN
    metrics["FP"] = FP
    metrics["FN"] = FN

    # Compute TPR, TNR, FPR, FNR.
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    metrics["TPR"] = TPR
    metrics["TNR"] = TNR
    metrics["FPR"] = FPR
    metrics["FNR"] = FNR

    # Compute accuracies.
    metrics["Accuracy"] = (TP + TN) / (TP + TN + FP + FN)
    metrics["Balanced Accuracy"] = (TPR + TNR) / 2

    # Compute F1 score.
    recall = TPR
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    metrics["F1 Score"] = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return metrics


def compute_detection_results(
    exp_key: str,
    quantile_key: str,
    results_dict: Dict[str, Dict[str, Any]],
    demo_results_frame: pd.DataFrame,
    test_results_frame: pd.DataFrame,
    detector: str,
    detector_kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """Compute detection results at timestep and trajectory level."""

    def store_result() -> None:
        result = {
            "timestep": {
                "metrics": None,
                "data": {
                    "calib_scores": calib_scores,
                    "test_scores": test_scores,
                    "test_preds": test_preds,
                    "test_labels": test_labels,
                },
            },
            "episode": {
                "metrics": None,
                "data": None,
            },
            "metadata": {
                "exp_key": exp_key,
                "quantile_key": quantile_key,
                "detector": detector,
                "detector_kwargs": detector_kwargs,
            },
        }

        # Timestep statistics.
        result["timestep"]["metrics"] = compute_metrics(
            preds=test_preds,
            labels=test_labels,
            scores=test_scores,
            success_labels=True,
        )

        # Compute episode statistics.
        metric = "cum_score" if "cum" in calib_key else "score"
        episode_labels = np.zeros(num_episodes, dtype=bool)
        episode_preds = np.zeros(num_episodes, dtype=bool)
        episode_scores = np.zeros(num_episodes, dtype=float)
        episode_detection_times = np.zeros(num_episodes, dtype=float)
        for i in range(num_episodes):
            episode_frame = data_utils.get_episode(
                test_results_frame, i, use_index=True
            )
            label = episode_frame.iloc[0].to_dict()["success"]
            pred = np.any(episode_frame[f"{quantile_key}_{calib_key}_pred"].values)
            if pred:
                pred_idx: int = np.where(
                    episode_frame[f"{quantile_key}_{calib_key}_pred"].values == True
                )[0][0]
            else:
                pred_idx = -1
            pred_score = episode_frame.iloc[pred_idx].to_dict()[f"{exp_key}_{metric}"]
            pred_time = episode_frame.iloc[pred_idx].to_dict()["timestep"]

            episode_labels[i] = label
            episode_preds[i] = pred
            episode_scores[i] = pred_score
            episode_detection_times[i] = pred_time

        result["episode"]["metrics"] = compute_metrics(
            preds=episode_preds,
            labels=episode_labels,
            scores=episode_scores,
            times=episode_detection_times,
            success_labels=True,
        )
        result["episode"]["data"] = {
            "calib_scores": calib_scores,
            "test_scores": episode_scores,
            "test_preds": episode_preds,
            "test_labels": episode_labels,
            "test_detection_times": episode_detection_times,
        }

        # Store result.
        results_dict[quantile_key][calib_key] = result

    def print_result() -> None:
        # Print result.
        dbprint(f"\nEpisode Results: {calib_key} | {quantile_key}")
        r: Dict[str, Any] = results_dict[quantile_key][calib_key]["episode"]["metrics"]
        dbprint(
            f"TPR: {r['TPR']:.2f} | TNR: {r['TNR']:.2f} | Acc: {r['Accuracy']:.2f} | Bal. Acc: {r['Balanced Accuracy']:.2f}"
        )
        dbprint(
            f"TP Time {r.get('TP Time Mean', -1.0):.2f} ({r.get('TP Time STD', -1.0):.2f})"
        )

    # General use arrays.
    num_episodes = data_utils.num_episodes(test_results_frame)
    demo_times: np.ndarray = np.unique(demo_results_frame["timestep"].values)
    test_times: np.ndarray = np.unique(test_results_frame["timestep"].values)
    test_labels = test_results_frame["success"].values

    ## Test scores for detectors that consider per-timestep errors.
    test_scores = test_results_frame[f"{exp_key}_score"].values

    # Method 1: I.I.D. timestep errors.
    calib_key = "t_iid"
    calib_scores = demo_results_frame[f"{exp_key}_score"].values
    test_preds = get_detector(detector)(calib_scores, test_scores, **detector_kwargs)
    test_results_frame = pd.concat(
        [
            test_results_frame,
            pd.Series(test_preds, name=f"{quantile_key}_{calib_key}_pred"),
        ],
        axis=1,
    )
    store_result()
    # print_result()

    # Method 2: I.I.D. episode errors.
    calib_key = "ep_iid_max"
    calib_scores = data_utils.aggr_episode_key_data(
        demo_results_frame, f"{exp_key}_score"
    )
    test_preds = get_detector(detector)(calib_scores, test_scores, **detector_kwargs)
    test_results_frame = pd.concat(
        [
            test_results_frame,
            pd.Series(test_preds, name=f"{quantile_key}_{calib_key}_pred"),
        ],
        axis=1,
    )
    store_result()
    # print_result()

    # Method 3: I.I.D. episode errors; per timestep calibration.
    calib_key = "ep_iid_per_t"
    test_results_frame = pd.concat(
        [
            test_results_frame,
            pd.Series(
                np.zeros(len(test_results_frame), dtype=bool),
                name=f"{quantile_key}_{calib_key}_pred",
            ),
        ],
        axis=1,
    )
    calib_scores = []
    for t in test_times:
        # Extract frames for timestep t.
        demo_t = t if t in demo_times else demo_times.max()
        demo_frames_t: pd.DataFrame = demo_results_frame[
            demo_results_frame["timestep"] <= demo_t
        ]
        test_frames_t: pd.DataFrame = test_results_frame[
            test_results_frame["timestep"] == t
        ]

        # Extract scores for timestep t.
        calib_scores.append(
            data_utils.aggr_episode_key_data(demo_frames_t, f"{exp_key}_score")
        )
        test_scores_t = test_frames_t[f"{exp_key}_score"].values

        # Compute and store predictions for timestep t.
        test_preds_t = get_detector(detector)(
            calib_scores[-1], test_scores_t, **detector_kwargs
        )
        test_results_frame.loc[
            test_results_frame["timestep"] == t, f"{quantile_key}_{calib_key}_pred"
        ] = test_preds_t

    test_preds = test_results_frame[f"{quantile_key}_{calib_key}_pred"].values
    store_result()
    # print_result()

    ## Test scores for detectors that consider cumulative errors.
    test_scores = test_results_frame[f"{exp_key}_cum_score"].values

    # Method 4: I.I.D. cumulative episode errors.
    calib_key = "ep_iid_cum"
    calib_scores = data_utils.aggr_episode_key_data(
        demo_results_frame, f"{exp_key}_cum_score"
    )
    test_preds = get_detector(detector)(calib_scores, test_scores, **detector_kwargs)
    test_results_frame = pd.concat(
        [
            test_results_frame,
            pd.Series(test_preds, name=f"{quantile_key}_{calib_key}_pred"),
        ],
        axis=1,
    )
    store_result()
    print_result()

    # I.I.D. episode cumalitive errors; per timestep calibration.
    calib_key = "ep_iid_cum_per_t"
    test_results_frame = pd.concat(
        [
            test_results_frame,
            pd.Series(
                np.zeros(len(test_results_frame), dtype=bool),
                name=f"{quantile_key}_{calib_key}_pred",
            ),
        ],
        axis=1,
    )
    calib_scores = []
    for t in test_times:
        # Extract frames for timestep t.
        demo_t = t if t in demo_times else demo_times.max()
        demo_frames_t: pd.DataFrame = demo_results_frame[
            demo_results_frame["timestep"] <= demo_t
        ]
        test_frames_t: pd.DataFrame = test_results_frame[
            test_results_frame["timestep"] == t
        ]

        # Extract scores for timestep t.
        calib_scores.append(
            data_utils.aggr_episode_key_data(demo_frames_t, f"{exp_key}_cum_score")
        )
        test_scores_t = test_frames_t[f"{exp_key}_cum_score"].values

        # Compute and store predictions for timestep t.
        test_preds_t = get_detector(detector)(
            calib_scores[-1], test_scores_t, **detector_kwargs
        )
        test_results_frame.loc[
            test_results_frame["timestep"] == t, f"{quantile_key}_{calib_key}_pred"
        ] = test_preds_t

    test_preds = test_results_frame[f"{quantile_key}_{calib_key}_pred"].values
    store_result()
    # print_result()

    return test_results_frame


def compute_prediction_results(
    exp_key: str,
    results_dict: Dict[str, Dict[str, Any]],
    test_results_frame: pd.DataFrame,
) -> None:
    """Compute detection results at timestep and trajectory level."""

    def store_result() -> None:
        result = {
            "timestep": {
                "metrics": None,
                "data": {
                    "test_preds": test_preds,
                    "test_labels": test_labels,
                },
            },
            "episode": {
                "metrics": None,
                "data": None,
            },
            "metadata": {
                "exp_key": exp_key,
            },
        }

        # Timestep statistics.
        result["timestep"]["metrics"] = compute_metrics(
            preds=test_preds,
            labels=test_labels,
            success_labels=True,
        )

        # Compute episode statistics.
        episode_labels = np.zeros(num_episodes, dtype=bool)
        episode_preds = np.zeros(num_episodes, dtype=bool)
        episode_detection_times = np.zeros(num_episodes, dtype=float)
        for i in range(num_episodes):
            episode_frame = data_utils.get_episode(
                test_results_frame, i, use_index=True
            )
            label = episode_frame.iloc[0].to_dict()["success"]
            pred = np.any(episode_frame[f"{exp_key}_pred"].values)
            if pred:
                pred_idx: int = np.where(
                    episode_frame[f"{exp_key}_pred"].values == True
                )[0][0]
            else:
                pred_idx = -1
            pred_time = episode_frame.iloc[pred_idx].to_dict()["timestep"]

            episode_labels[i] = label
            episode_preds[i] = pred
            episode_detection_times[i] = pred_time

        result["episode"]["metrics"] = compute_metrics(
            preds=episode_preds,
            labels=episode_labels,
            times=episode_detection_times,
            success_labels=True,
        )
        result["episode"]["data"] = {
            "test_preds": episode_preds,
            "test_labels": episode_labels,
            "test_detection_times": episode_detection_times,
        }

        # Store result.
        results_dict[exp_key] = result

    def print_result() -> None:
        # Print result.
        dbprint(f"\nEpisode Results: {exp_key}")
        r: Dict[str, Any] = results_dict[exp_key]["episode"]["metrics"]
        dbprint(
            f"TPR: {r['TPR']:.2f} | TNR: {r['TNR']:.2f} | Acc: {r['Accuracy']:.2f} | Bal. Acc: {r['Balanced Accuracy']:.2f}"
        )
        dbprint(
            f"TP Time {r.get('TP Time Mean', -1.0):.2f} ({r.get('TP Time STD', -1.0):.2f})"
        )

    # General use arrays.
    num_episodes = data_utils.num_episodes(test_results_frame)
    test_labels = test_results_frame["success"].values
    test_preds = test_results_frame[f"{exp_key}_pred"]
    store_result()
    print_result()

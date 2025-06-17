from typing import Sequence

import numpy as np
from scipy import spatial
from sklearn import metrics
import pandas as pd
from condorgmm.utils.common.pose import Pose
from concurrent.futures import ThreadPoolExecutor


def apply_transform(pose: np.ndarray, vertices: np.ndarray):
    return (pose[:3, :3] @ vertices.T + pose[:3, 3][:, None]).T


def add_err(pred_pose: np.ndarray, gt_pose: np.ndarray, vertices: np.ndarray):
    pred_locs = apply_transform(pred_pose, vertices)
    gt_locs = apply_transform(gt_pose, vertices)
    return np.linalg.norm(pred_locs - gt_locs, axis=-1).mean()


def adds_err(pred_pose: np.ndarray, gt_pose: np.ndarray, vertices: np.ndarray):
    pred_locs = apply_transform(pred_pose, vertices)
    gt_locs = apply_transform(gt_pose, vertices)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pred_locs)
    nn_dists, _ = nn_index.query(gt_locs, k=1)

    return nn_dists.mean()


ALL_OBJECT_POSE_METRICS = {
    "ADD-S": adds_err,
    "ADD": add_err,
}


def create_empty_results_dataframe():
    df = pd.DataFrame(
        columns=[
            "scene",
            "method",
            "object",
            "timestep",
            "predicted",
            "gt",
            "metric",
            "value",
        ]
    )
    return df


def add_object_tracking_metrics_to_results_dataframe(
    results_df,
    scene_id,
    method,
    object_name,
    predicted_poses,
    gt_poses,
    vertices,
    metrics=ALL_OBJECT_POSE_METRICS,
    other_info={},
):
    assert len(predicted_poses) == len(gt_poses)
    for metric_name, metric_fn in ALL_OBJECT_POSE_METRICS.items():
        with ThreadPoolExecutor() as executor:
            metric_values = list(
                executor.map(
                    lambda i: metric_fn(
                        Pose(predicted_poses[i]).as_matrix(),
                        Pose(gt_poses[i]).as_matrix(),
                        vertices,
                    ),
                    range(len(predicted_poses)),
                )
            )

        for timestep, metric_value in enumerate(metric_values):
            results_df.loc[len(results_df)] = {
                "scene": scene_id,
                "method": method,
                "object": object_name,
                "timestep": timestep,
                "metric": metric_name,
                "predicted": predicted_poses[timestep].posquat,
                "gt": gt_poses[timestep].posquat,
                "value": metric_value,
                **other_info,
            }


# def add_foundation_pose_object_tracking_metrics_to_results_dataframe(
#     ycb_dir,
#     results_df,
#     scene_id,
#     FRAME_RATE,
#     object_index,
#     timesteps=None,
# ) -> pd.DataFrame:
#     fp_loader = YCBVTrackingResultLoader(frame_rate=FRAME_RATE, split=ycb_dir.name)
#     fp_accuracy_metrics = fp_loader.get_metrics(scene_id, object_index)
#     fp_poses = Pose.from_matrix(fp_loader.get_poses(scene_id, object_index))
#     gt_poses = Pose.from_matrix(fp_loader.get_gt_poses(scene_id, object_index))
#     num_frames = len(fp_poses)
#     if timesteps is None:
#         timesteps = range(num_frames)
#     else:
#         gt_poses = gt_poses[timesteps]
#         fp_poses = fp_poses[timesteps]

#     # we are storing one and only one object per file and we can obtain the
#     # object name out of it
#     any_metric = next(iter(ALL_OBJECT_POSE_METRICS))
#     assert len(fp_accuracy_metrics[any_metric]) == 1
#     object_name = next(iter(fp_accuracy_metrics[any_metric]))

#     for metric_name in ALL_OBJECT_POSE_METRICS.keys():
#         metric_values = np.array(fp_accuracy_metrics[metric_name][object_name])
#         metric_values = metric_values[timesteps]

#         df = pd.DataFrame.from_dict(
#             {
#                 "scene": scene_id,
#                 "method": "FoundationPose",
#                 "object": object_name,
#                 "timestep": timesteps,
#                 "predicted": list(fp_poses.posquat),
#                 "gt": list(gt_poses.posquat),
#                 "metric": metric_name,
#                 "value": metric_values,
#             }
#         )
#         results_df = pd.concat([results_df, df]) if not results_df.empty else df
#     return results_df


#### Camera Pose Tracking Metrics


def align(model, data):
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3, -1))
    data_zerocentered = data - data.mean(1).reshape((3, -1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1).reshape((3, -1)) - rot * model.mean(1).reshape((3, -1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def evaluate_ate(gt_poses, predicted_poses):
    assert len(gt_poses) == len(predicted_poses)
    gt_traj_pts = [gt_poses[idx].pos for idx in range(len(gt_poses))]
    est_traj_pts = [predicted_poses[idx].pos for idx in range(len(predicted_poses))]

    gt_traj_pts = np.stack(gt_traj_pts).T
    est_traj_pts = np.stack(est_traj_pts).T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)
    return trans_error


def add_camera_tracking_metrics_to_results_dataframe(
    results_df,
    scene_id,
    method,
    predicted_poses,
    gt_poses,
):
    ate = evaluate_ate(gt_poses, predicted_poses)
    for timestep in range(len(gt_poses)):
        results_df.loc[len(results_df)] = {
            "scene": scene_id,
            "method": method,
            "object": "camera",
            "timestep": timestep,
            "metric": "ATE",
            "predicted": np.array(predicted_poses[timestep].posquat),
            "gt": np.array(gt_poses[timestep].posquat),
            "value": ate[timestep],
        }


def compute_auc(errs: Sequence, max_val: float = 0.1, step=0.001):
    errs = np.sort(np.array(errs))
    X = np.arange(0, max_val + step, step)
    Y = np.ones(len(X))
    for i, x in enumerate(X):
        y = (errs <= x).sum() / len(errs)
        Y[i] = y
        if y >= 1:
            break
    auc = metrics.auc(X, Y) / (max_val * 1)
    return auc


def aggregate_dataframe_with_function(results_df, func):
    return results_df.groupby(["scene", "metric", "object", "method"])["value"].apply(
        func
    )

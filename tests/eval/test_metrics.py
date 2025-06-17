from condorgmm.eval.metrics import evaluate_ate
import condorgmm.eval.metrics
from condorgmm import Pose
import numpy as np


def test_evaluate_ate():
    # Random gt poses
    num_poses = 100
    gt_poses = Pose(np.random.rand(num_poses, 3))
    # Random predicted poses
    predicted_poses = Pose(np.random.rand(num_poses, 3))
    # Evaluate ATE
    ate = evaluate_ate(gt_poses, gt_poses)
    print(ate)
    assert np.isclose(ate.mean(), 0.0)

    ate = evaluate_ate(predicted_poses, predicted_poses)
    print(ate)
    assert np.isclose(ate.mean(), 0.0)

    ate = evaluate_ate(gt_poses, Pose(gt_poses.posquat + 0.5))
    print(ate)
    assert np.isclose(ate.mean(), 0.0)

    ate = evaluate_ate(gt_poses, predicted_poses)
    print(ate)
    assert np.all(ate >= 0.0)


def test_aggregate_camera_tracking_metrics():
    results_df = condorgmm.eval.metrics.create_empty_results_dataframe()

    # Random gt poses
    num_poses = 100
    gt_poses = Pose(np.random.rand(num_poses, 3))
    # Random predicted poses
    predicted_poses = Pose(np.random.rand(num_poses, 3))

    condorgmm.eval.metrics.add_camera_tracking_metrics_to_results_dataframe(
        results_df,
        "0",
        "condorgmm",
        gt_poses,
        predicted_poses,
    )
    print(results_df)

    grouped_df = results_df.groupby(["metric", "object", "method"])["value"]

    print(grouped_df)
    print(grouped_df.apply(np.mean))

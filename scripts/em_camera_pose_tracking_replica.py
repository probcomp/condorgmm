import condorgmm
import condorgmm.warp_gmm as warp_gmm
import condorgmm.eval.metrics
import condorgmm.data
from tqdm import tqdm
import importlib
import warp as wp
import numpy as np
import condorgmm.warp_gmm.kernels


import condorgmm
import condorgmm.utils.common.rerun

import fire


def run_replica(
    scene=None,
    window_half_width=4,
    stride=100,
    downscale=1,
    update_every=10,
):
    if scene is None:
        scene = "office1"

    condorgmm.rr_init(f"camera_pose_tracking_{scene}")
    print("Running for scene: ", scene)
    video = condorgmm.data.ReplicaVideo(scene).downscale(downscale)
    importlib.reload(condorgmm.data.base_dataloading)
    frames = video.load_all_frames()

    VIZ_LOG_SCORE_IMAGE = False
    t = 0
    frame = frames[t]
    frame_warp = frame.as_warp()

    camera_pose = condorgmm.Pose(frame.camera_pose)

    STRIDE = stride
    spatial_means = condorgmm.xyz_from_depth_image(
        frame.depth.astype(np.float32), *frame.intrinsics
    )[::STRIDE, ::STRIDE].reshape(-1, 3)
    spatial_means = camera_pose.apply(spatial_means).astype(np.float32)
    rgb_means = frame.rgb[::STRIDE, ::STRIDE].reshape(-1, 3).astype(np.float32)

    gmm = warp_gmm.gmm_warp_from_numpy(spatial_means, rgb_means)
    gmm.camera_posquat = wp.array(camera_pose.posquat.astype(np.float32))

    warp_gmm_state = warp_gmm.initialize_state(gmm=gmm, frame=frame)
    warp_gmm_state.hyperparams.window_half_width = 5

    condorgmm.rr_set_time(-2)
    warp_gmm.rr_log_gmm_warp(warp_gmm_state.gmm)
    warp_gmm.warp_gmm_forward(frame_warp, warp_gmm_state)
    condorgmm.rr_log_depth(warp_gmm_state.log_score_image.numpy())

    for _ in range(5):
        warp_gmm.warp_gmm_EM_step(frame_warp, warp_gmm_state)

    condorgmm.rr_set_time(-1)
    warp_gmm.rr_log_gmm_warp(warp_gmm_state.gmm)
    condorgmm.rr_log_depth(warp_gmm_state.log_score_image.numpy())
    assert warp_gmm_state.gmm.is_valid()

    all_data = {}
    camera_poses_inferred = {}
    camera_poses_inferred[t - 1] = warp_gmm_state.gmm.camera_posquat.numpy()
    camera_poses_inferred[t - 2] = warp_gmm_state.gmm.camera_posquat.numpy()
    gmms = {}

    warp_gmm_state.gmm.camera_posquat.requires_grad = True

    learning_rates = wp.array(
        [0.002, 0.002, 0.002, 0.0004, 0.0004, 0.0004, 0.0004], dtype=wp.float32
    )

    for t in tqdm(range(t, len(frames))):
        condorgmm.rr_set_time(t)
        frame = frames[t]
        frame_warp = frame.as_warp()

        prev_prev_pose = camera_poses_inferred[t - 2]
        prev_pose = camera_poses_inferred[t - 1]
        prev_prev_quat = prev_prev_pose[3:]
        prev_quat = prev_pose[3:]

        prev_prev_quat = prev_prev_quat / np.linalg.norm(prev_prev_quat)
        prev_quat = prev_quat / np.linalg.norm(prev_quat)

        new_quat = prev_quat + (prev_quat - prev_prev_quat)
        new_quat = new_quat / np.linalg.norm(new_quat)

        new_pose = np.array(
            [*(prev_pose[:3] + (prev_pose[:3] - prev_prev_pose[:3])), *(new_quat)]
        )

        warp_gmm_state.gmm.camera_posquat = wp.array(new_pose.astype(np.float32))
        warp_gmm_state.gmm.camera_posquat.requires_grad = True

        data = warp_gmm.optimize_params(
            [warp_gmm_state.gmm.camera_posquat],
            frame_warp,
            warp_gmm_state,
            199,
            learning_rates,
            storing_stuff=False,
        )
        all_data[t] = data
        condorgmm.rr_log_posquat(frame.camera_pose, channel="gt_pose")
        condorgmm.rr_log_posquat(warp_gmm_state.gmm.camera_posquat.numpy(), channel="pose")
        # condorgmm.rr_log_frame(frame, camera_pose=frame.camera_pose)

        camera_poses_inferred[t] = warp_gmm_state.gmm.camera_posquat.numpy()

        if VIZ_LOG_SCORE_IMAGE:
            condorgmm.rr_log_depth(warp_gmm_state.log_score_image.numpy())

        if t > 0 and t % update_every == 0:
            camera_pose = condorgmm.Pose(warp_gmm_state.gmm.camera_posquat.numpy())
            spatial_means = condorgmm.xyz_from_depth_image(
                frame.depth.astype(np.float32), *frame.intrinsics
            )[::STRIDE, ::STRIDE].reshape(-1, 3)
            spatial_means = camera_pose.apply(spatial_means).astype(np.float32)
            rgb_means = frame.rgb[::STRIDE, ::STRIDE].reshape(-1, 3).astype(np.float32)

            gmm = warp_gmm.gmm_warp_from_numpy(spatial_means, rgb_means)
            gmm.camera_posquat = wp.array(camera_pose.posquat.astype(np.float32))
            warp_gmm_state.gmm = gmm

            for _ in range(5):
                warp_gmm.warp_gmm_EM_step(frame_warp, warp_gmm_state)
            warp_gmm.rr_log_gmm_warp(warp_gmm_state.gmm)
            warp_gmm_state.gmm.camera_posquat.requires_grad = True
            gmms[t] = warp_gmm_state.gmm

    final_T = len(frames)
    gt_poses = [condorgmm.Pose(frame.camera_pose) for frame in frames[:final_T]]
    predicted_poses = [condorgmm.Pose(camera_poses_inferred[t]) for t in range(final_T)]
    ate = condorgmm.eval.metrics.evaluate_ate(gt_poses, predicted_poses)
    print(f"ATE: {ate}")
    print(f"ATE avg: {ate.mean()}")


if __name__ == "__main__":
    fire.Fire(run_replica)

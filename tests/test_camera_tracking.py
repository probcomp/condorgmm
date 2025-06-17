import condorgmm
import condorgmm.eval.metrics
import condorgmm.data
from tqdm import tqdm
import condorgmm.camera_tracking.integrated_camera_tracking
import importlib

import condorgmm.warp_gmm as warp_gmm
import warp as wp
import numpy as np
import matplotlib.pyplot as plt


def test_one_step():
    wp.init()

    scene = "office1"
    video = condorgmm.data.ReplicaVideo(scene).downscale(10)
    importlib.reload(condorgmm.data.base_dataloading)
    max_T_local = 10
    frames = [video[i] for i in tqdm(range(0, max_T_local, 1))]

    _, state = condorgmm.camera_tracking.integrated_camera_tracking.initialize(frames[0])
    importlib.reload(condorgmm.camera_tracking.integrated_camera_tracking)
    _, state = condorgmm.camera_tracking.integrated_camera_tracking.update(state, frames[0])


def test_optimize_camera_pose_at_one_frame():
    wp.init()

    scene = "office1"
    video = condorgmm.data.ReplicaVideo(scene).downscale(10)
    importlib.reload(condorgmm.data.base_dataloading)
    max_T_local = 10
    frames = [video[i] for i in tqdm(range(0, max_T_local, 1))]
    frame = frames[0]
    _, state = condorgmm.camera_tracking.integrated_camera_tracking.initialize(frame)

    warp_gmm_state = state.warp_gmm_state

    warp_gmm_state.camera_posquat = wp.from_jax(state.camera_pose.posquat)
    warp_gmm_state.camera_posquat.requires_grad = True

    frame_warp = frames[0].as_warp()
    params_over_time, likelihood_over_time = warp_gmm.optimize_params(
        [warp_gmm_state.camera_posquat],
        frame_warp,
        warp_gmm_state,
        num_timesteps=100,
        lr=1e-3,
    )

    plt.plot(likelihood_over_time)
    plt.savefig("test_optimize_camera_pose_at_one_frame.png")

    print("Difference ", params_over_time[-1][0] - params_over_time[0][0])

    condorgmm.rr_init("camera_tracking_test")

    condorgmm.rr_log_posquat(frame.camera_pose, channel="gt_pose")
    for t in range(len(params_over_time)):
        condorgmm.rr_set_time(t)
        condorgmm.rr_log_posquat(params_over_time[t][0], channel="pose")


def test_optimize_camera_pose_denser_gmm():
    scene = "office1"
    video = condorgmm.data.ReplicaVideo(scene).downscale(10)
    importlib.reload(condorgmm.data.base_dataloading)
    max_T_local = 10
    frames = [video[i] for i in tqdm(range(0, max_T_local, 1))]
    frame = frames[0]

    initial_camera_pose = condorgmm.Pose(frame.camera_pose)

    STRIDE = 5
    spatial_means = condorgmm.xyz_from_depth_image(
        frame.depth.astype(np.float32), *frame.intrinsics
    )[::STRIDE, ::STRIDE].reshape(-1, 3)
    spatial_means = initial_camera_pose.apply(spatial_means).astype(np.float32)
    rgb_means = frame.rgb[::STRIDE, ::STRIDE].reshape(-1, 3).astype(np.float32)

    log_scales_xyz = np.log(
        np.tile(
            np.array([0.01, 0.01, 0.01], dtype=np.float32), (spatial_means.shape[0], 1)
        )
    )
    log_scales_rgb = np.log(
        np.tile(np.array([5.1, 5.1, 5.1], dtype=np.float32), (rgb_means.shape[0], 1))
    )

    importlib.reload(warp_gmm)
    gmm = warp_gmm.GMM_Warp.create_from_numpy(
        spatial_means, rgb_means, log_scales_xyz, log_scales_rgb
    )

    warp_gmm_state = warp_gmm.initialize_state(gmm, frame)
    frame_warp = frame.as_warp()

    warp_gmm_state.camera_posquat = wp.array(
        initial_camera_pose.posquat.astype(np.float32)
    )

    warp_gmm_state.camera_posquat.requires_grad = True
    params_over_time, likelihood_over_time = warp_gmm.optimize_params(
        [warp_gmm_state.camera_posquat],
        frame_warp,
        warp_gmm_state,
        num_timesteps=100,
        lr=5e-4,
    )
    plt.plot(likelihood_over_time)
    plt.savefig("test_optimize_camera_pose_denser_gmm.png")

    print("Difference ", params_over_time[-1][0] - params_over_time[0][0])
    condorgmm.rr_init("camera_tracking_test")

    condorgmm.rr_log_posquat(frame.camera_pose, channel="gt_pose")
    for t in range(len(params_over_time)):
        condorgmm.rr_set_time(t)
        condorgmm.rr_log_posquat(params_over_time[t][0], channel="pose")

import numpy as np
import warp as wp
import condorgmm.warp_gmm as warp_gmm
from tqdm import tqdm
import condorgmm
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def test_optimize():
    scene = "office1"
    video = condorgmm.data.ReplicaVideo(scene)
    max_T_local = 10
    frames = [video[i] for i in tqdm(range(0, max_T_local, 1))]
    frame = frames[0]

    initial_camera_pose = condorgmm.Pose(frame.camera_pose)

    STRIDE = 3
    spatial_means = condorgmm.xyz_from_depth_image(
        frame.depth.astype(np.float32), *frame.intrinsics
    )[::STRIDE, ::STRIDE].reshape(-1, 3)
    spatial_means = initial_camera_pose.apply(spatial_means).astype(np.float32)
    rgb_means = frame.rgb[::STRIDE, ::STRIDE].reshape(-1, 3).astype(np.float32)

    scales_xyz = np.log(
        np.tile(
            np.array([0.01, 0.01, 0.01], dtype=np.float32), (spatial_means.shape[0], 1)
        )
    )
    scales_rgb = np.log(
        np.tile(np.array([5.1, 5.1, 5.1], dtype=np.float32), (rgb_means.shape[0], 1))
    )

    gmm = warp_gmm.gmm_warp_from_numpy(spatial_means, rgb_means, scales_xyz, scales_rgb)

    warp_gmm_state = warp_gmm.initialize_state(gmm, frame)
    frame_warp = frame.as_warp()

    warp_gmm_state.gmm.camera_posquat = wp.array(
        initial_camera_pose.posquat.astype(np.float32)
        + np.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    )

    warp_gmm_state.gmm.camera_posquat.requires_grad = True
    camera_poses_over_time, likelihood_over_time = warp_gmm.optimize_params(
        [warp_gmm_state.gmm.camera_posquat],
        frame_warp,
        warp_gmm_state,
        num_timesteps=100,
        lr=1e-3,
    )
    camera_poses_over_time = [i[0] for i in camera_poses_over_time]

    plt.matshow(warp_gmm_state.log_score_image.numpy())
    plt.colorbar()
    plt.savefig("test_optimize_log_score_image.png")

    condorgmm.rr_init("test_optimize")

    condorgmm.rr_log_posquat(frame.camera_pose, channel="gt_pose")
    for t in range(len(camera_poses_over_time)):
        condorgmm.rr_set_time(t)
        condorgmm.rr_log_posquat(camera_poses_over_time[t], channel="pose")


def test_optimize_through_jax():
    scene = "office1"
    video = condorgmm.data.ReplicaVideo(scene)
    max_T_local = 10
    frames = [video[i] for i in tqdm(range(0, max_T_local, 1))]
    frame = frames[0]

    initial_camera_pose = condorgmm.Pose(frame.camera_pose)

    STRIDE = 2
    spatial_means = condorgmm.xyz_from_depth_image(
        frame.depth.astype(np.float32), *frame.intrinsics
    )[::STRIDE, ::STRIDE].reshape(-1, 3)
    spatial_means = initial_camera_pose.apply(spatial_means).astype(np.float32)
    rgb_means = frame.rgb[::STRIDE, ::STRIDE].reshape(-1, 3).astype(np.float32)

    scales_xyz = np.log(
        np.tile(
            np.array([0.01, 0.01, 0.01], dtype=np.float32), (spatial_means.shape[0], 1)
        )
    )
    scales_rgb = np.log(
        np.tile(np.array([5.1, 5.1, 5.1], dtype=np.float32), (rgb_means.shape[0], 1))
    )

    gmm = warp_gmm.gmm_warp_from_numpy(spatial_means, rgb_means, scales_xyz, scales_rgb)

    warp_gmm_state = warp_gmm.initialize_state(gmm, frame)
    frame_warp = frame.as_warp()

    warp_gmm_state.camera_posquat = wp.array(
        initial_camera_pose.posquat.astype(np.float32)
        + np.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    )
    warp_gmm_state.camera_posquat.requires_grad = True

    # Convert warp array to jax array for optimization
    camera_posquat_jax = jnp.array(warp_gmm_state.camera_posquat.numpy())

    # Create optimizer
    optimizer = optax.adam(learning_rate=5e-3)
    opt_state = optimizer.init(camera_posquat_jax)

    warp_gmm_state.camera_posquat = wp.from_jax(camera_posquat_jax)
    warp_gmm_state.camera_posquat.requires_grad = True

    # Optimization loop
    camera_poses_over_time = []
    for step in range(100):
        tape = wp.Tape()
        with tape:
            warp_gmm.warp_gmm_forward(
                frame_warp,
                warp_gmm_state,
            )
        tape.backward(grads={warp_gmm_state.log_score_image: warp_gmm_state.backward})

        grads = wp.to_jax(warp_gmm_state.camera_posquat.grad)
        updates, opt_state = optimizer.update(grads, opt_state)
        camera_posquat_jax = optax.apply_updates(camera_posquat_jax, updates)
        warp_gmm_state.camera_posquat = wp.from_jax(camera_posquat_jax)
        warp_gmm_state.camera_posquat.requires_grad = True
        tape.zero()
        wp.synchronize()

        camera_poses_over_time.append(warp_gmm_state.camera_posquat.numpy())

    condorgmm.rr_init("test_optimize_through_jax")

    condorgmm.rr_log_posquat(frame.camera_pose, channel="gt_pose")
    for t in range(len(camera_poses_over_time)):
        condorgmm.rr_set_time(t)
        condorgmm.rr_log_posquat(camera_poses_over_time[t], channel="pose")

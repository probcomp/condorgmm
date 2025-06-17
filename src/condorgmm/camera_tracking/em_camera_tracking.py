from condorgmm import Pose
from condorgmm.data import Frame
import condorgmm.warp_gmm as warp_gmm
import warp as wp
import condorgmm
import numpy as np

STRIDE = 100
UPDATE_EVERY = 10

learning_rates = wp.array(
    [0.002, 0.002, 0.002, 0.0004, 0.0004, 0.0004, 0.0004], dtype=wp.float32
)


def initialize(frame: Frame, debug=False, seed=0):
    frame_warp = frame.as_warp()
    camera_pose = condorgmm.Pose(frame.camera_pose)

    spatial_means = condorgmm.xyz_from_depth_image(
        frame.depth.astype(np.float32), *frame.intrinsics
    )[::STRIDE, ::STRIDE].reshape(-1, 3)
    spatial_means = camera_pose.apply(spatial_means).astype(np.float32)
    rgb_means = frame.rgb[::STRIDE, ::STRIDE].reshape(-1, 3).astype(np.float32)

    gmm = warp_gmm.gmm_warp_from_numpy(spatial_means, rgb_means)
    gmm.camera_posquat = wp.array(camera_pose.posquat.astype(np.float32))

    warp_gmm_state = warp_gmm.initialize_state(gmm=gmm, frame=frame)
    warp_gmm_state.hyperparams.window_half_width = 5

    warp_gmm.warp_gmm_forward(frame_warp, warp_gmm_state)

    for _ in range(5):
        warp_gmm.warp_gmm_EM_step(frame_warp, warp_gmm_state)

    assert warp_gmm_state.gmm.is_valid()

    two_prev_camera_poses = (
        warp_gmm_state.gmm.camera_posquat.numpy(),
        warp_gmm_state.gmm.camera_posquat.numpy(),
    )
    state = (
        warp_gmm_state,
        two_prev_camera_poses,
    )
    return camera_pose, state, {}


def update(
    state,
    frame: Frame,
    timestep,
    debug=False,
):
    warp_gmm_state, two_prev_camera_poses = state
    prev_prev_pose, prev_pose = two_prev_camera_poses
    frame_warp = frame.as_warp()

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

    _ = warp_gmm.optimize_params(
        [warp_gmm_state.gmm.camera_posquat],
        frame_warp,
        warp_gmm_state,
        99,
        learning_rates,
        storing_stuff=False,
    )

    if timestep > 0 and timestep % UPDATE_EVERY == 0:
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
        # warp_gmm.rr_log_gmm_warp(warp_gmm_state.gmm)

    inferred_camera_pose = condorgmm.Pose(warp_gmm_state.gmm.camera_posquat.numpy())

    debug_data = {}
    debug_data["gmm_is_valid"] = warp_gmm_state.gmm.is_valid()

    state = (warp_gmm_state, (prev_pose, inferred_camera_pose.posquat))

    return inferred_camera_pose, state, debug_data


def rr_log(
    state, frame: Frame, timestep: int, do_log_poses=True, do_log_frame=True, **kwargs
):
    condorgmm.rr_set_time(timestep)
    warp_gmm_state = state[0]
    inferred_camera_pose = condorgmm.Pose(warp_gmm_state.gmm.camera_posquat.numpy())
    condorgmm.rr_log_frame(
        frame.downscale(5),
        "observed_data_unprojected_from_inferred_camera_frame",
        camera_pose=inferred_camera_pose,
    )
    warp_gmm.rr_log_gmm_warp(warp_gmm_state.gmm, size_scalar=16.0)
    condorgmm.rr_log_pose(inferred_camera_pose, "inferred_camera_pose", scale=0.3)
    
    # if do_log_poses:
    #     condorgmm.rr_log_pose(Pose(frame.camera_pose), "true_camera_pose", scale=0.3)
    # if do_log_frame:
    #     # condorgmm.rr_log_frame(
    #     #     frame, "observed_data", camera_pose=condorgmm.Pose(frame.camera_pose)
    #     # )
    #     condorgmm.rr_log_frame(
    #         frame,
    #         "observed_data_unprojected_from_inferred_camera_frame",
    #         camera_pose=inferred_camera_pose,
    #     )

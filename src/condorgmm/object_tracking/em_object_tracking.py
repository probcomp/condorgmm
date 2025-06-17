from condorgmm.data import Frame
import condorgmm.warp_gmm as warp_gmm
import warp as wp
import condorgmm
import numpy as np
from trimesh import Trimesh

STRIDE = 100
UPDATE_EVERY = 10

learning_rates = wp.array(
    [0.002, 0.002, 0.002, 0.0004, 0.0004, 0.0004, 0.0004], dtype=wp.float32
)


def initialize(
    frame: Frame,
    object_mesh: Trimesh,
    object_idx=0,
    debug=False,
    seed=0,
):
    object_index = object_idx
    mask = frame.masks[object_index] * frame.depth > 0.001
    camera_pose = condorgmm.Pose(frame.camera_pose)
    object_pose = condorgmm.Pose(frame.object_poses[object_index])

    STRIDE = 5
    mask_strided = mask[::STRIDE, ::STRIDE]
    xyz = condorgmm.xyz_from_depth_image(frame.depth, *frame.intrinsics)
    xyz = camera_pose.apply(xyz)
    xyz_strided = xyz[::STRIDE, ::STRIDE]
    rgb_strided = frame.rgb[::STRIDE, ::STRIDE]
    spatial_means = xyz_strided[mask_strided]
    rgb_means = rgb_strided[mask_strided]

    warp_gmm_state = warp_gmm.initialize_state(frame=frame)

    gmm = warp_gmm.gmm_warp_from_numpy(
        object_pose.inv().apply(spatial_means).astype(np.float32),
        rgb_means.astype(np.float32),
    )
    warp_gmm_state = warp_gmm.initialize_state(frame=frame)
    warp_gmm_state.gmm = gmm
    warp_gmm_state.gmm.object_posquats = wp.array(
        object_pose.posquat[None, ...], dtype=wp.float32
    )
    warp_gmm_state.gmm.camera_posquat = wp.array(camera_pose.posquat, dtype=wp.float32)
    frame_warp = frame.as_warp()
    warp_gmm.warp_gmm_forward(frame_warp, warp_gmm_state)

    full_mask = wp.array(
        np.ones((frame.height, frame.width), dtype=wp.bool), dtype=wp.bool
    )
    partial_mask = wp.array(mask, dtype=wp.bool)

    warp_gmm_state.mask = partial_mask
    for _ in range(10):
        warp_gmm.warp_gmm_EM_step(frame_warp, warp_gmm_state)
    warp_gmm_state.mask = full_mask

    warp_gmm.warp_gmm_forward(frame_warp, warp_gmm_state)
    log_score_image_after = warp_gmm_state.log_score_image.numpy()

    condorgmm.rr_log_frame(frame)
    condorgmm.rr_log_depth(log_score_image_after, channel="frame/log_score_image")

    inferred_camera_pose = condorgmm.Pose(warp_gmm_state.gmm.camera_posquat.numpy())
    inferred_object_pose = condorgmm.Pose(warp_gmm_state.gmm.object_posquats.numpy()[0])
    return inferred_camera_pose, inferred_object_pose, (warp_gmm_state, object_idx), {}


def update(
    state,
    frame: Frame,
    timestep: int,
    debug=False,
):
    warp_gmm_state, _ = state
    frame_warp = frame.as_warp()
    warp_gmm_state.gmm.object_posquats.requires_grad = True
    warp_gmm_state.gmm.camera_posquat.requires_grad = True
    _ = warp_gmm.optimize_params(
        [
            warp_gmm_state.gmm.object_posquats,
        ],
        frame_warp,
        warp_gmm_state,
        200,
        learning_rates,
        storing_stuff=False,
    )
    # warp_gmm.rr_log_gmm_warp(warp_gmm_state.gmm)
    inferred_camera_pose = condorgmm.Pose(warp_gmm_state.gmm.camera_posquat.numpy())
    inferred_object_pose = condorgmm.Pose(warp_gmm_state.gmm.object_posquats.numpy()[0])
    return inferred_camera_pose, inferred_object_pose, state, {}


def rr_log(
    state,
    frame: Frame,
    timestep: int,
    do_log_poses=True,
    do_log_frame=True,
):
    condorgmm.rr_set_time(timestep)
    warp_gmm_state, object_index = state


    gt_object_pose = condorgmm.Pose(frame.object_poses[object_index])
    gt_camera_pose = condorgmm.Pose(frame.camera_pose)
    gt_object_pose_in_camera_frame = gt_camera_pose.inv() @ gt_object_pose

    inferred_object_pose = condorgmm.Pose(warp_gmm_state.gmm.object_posquats.numpy()[0])
    inferred_camera_pose = condorgmm.Pose(warp_gmm_state.gmm.camera_posquat.numpy())
    inferred_object_pose_in_camera_frame = (
        inferred_camera_pose.inv() @ inferred_object_pose
    )

    warp_gmm.rr_log_gmm_warp(warp_gmm_state.gmm, size_scalar=2.0)
    condorgmm.rr_log_pose(
        inferred_object_pose, "inferred_object_pose_in_camera_frame", radii=0.001
    )
    # condorgmm.rr_log_pose(
    #     gt_object_pose_in_camera_frame, "true_object_pose_in_camera_frame"
    # )

    condorgmm.rr_log_frame(
        frame,
        "observed_data_unprojected_from_inferred_camera_frame",
        camera_pose=inferred_camera_pose,
    )


    # if do_log_poses:
    #     gt_object_pose = condorgmm.Pose(frame.object_poses[object_index])
    #     gt_camera_pose = condorgmm.Pose(frame.camera_pose)
    #     gt_object_pose_in_camera_frame = gt_camera_pose.inv() @ gt_object_pose

    #     inferred_object_pose = condorgmm.Pose(warp_gmm_state.gmm.object_posquats.numpy()[0])
    #     inferred_camera_pose = condorgmm.Pose(warp_gmm_state.gmm.camera_posquat.numpy())
    #     inferred_object_pose_in_camera_frame = (
    #         inferred_camera_pose.inv() @ inferred_object_pose
    #     )

    #     condorgmm.rr_log_pose(
    #         inferred_object_pose_in_camera_frame, "inferred_object_pose_in_camera_frame"
    #     )
    #     condorgmm.rr_log_pose(
    #         gt_object_pose_in_camera_frame, "true_object_pose_in_camera_frame"
    #     )
    # if do_log_frame:
    #     condorgmm.rr_log_frame(
    #         frame, "observed_data", camera_pose=condorgmm.Pose(frame.camera_pose)
    #     )
    #     condorgmm.rr_log_frame(
    #         frame,
    #         "observed_data_unprojected_from_inferred_camera_frame",
    #         camera_pose=inferred_camera_pose,
    #     )

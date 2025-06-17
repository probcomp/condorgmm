from condorgmm.data import Frame
import condorgmm.warp_gmm as warp_gmm
import warp as wp
import condorgmm
import numpy as np
from trimesh import Trimesh
from condorgmm.warp_gmm.enumeration_kernels import inference_step
import importlib
import condorgmm
import condorgmm.data
import matplotlib.pyplot as plt
import warp as wp
import numpy as np
import scipy.stats
import condorgmm.warp_gmm as warp_gmm
import trimesh

STRIDE = 100
UPDATE_EVERY = 10

learning_rates = wp.array(
    [0.002, 0.002, 0.002, 0.0004, 0.0004, 0.0004, 0.0004], dtype=wp.float32
)

pose_hypotheses = None
pixel_coordinates = None
corresponding_rgbd_per_pose_and_vertex = None
scores_per_pose_and_vertex = None
scores_per_pose = None


num_poses = 20000

c2f_schedule_params = (
    (0.04, 1000.0),
    (0.02, 1500.0),
    (0.01, 3000.0),
    (0.005, 4000.0),
)
c2f_schedule = []

for c2f_step in c2f_schedule_params:
    sigma, kappa = c2f_step
    position_deltas = np.random.normal(0.0, sigma, size=(num_poses, 3))
    quaternion_deltas = scipy.stats.vonmises_fisher(
        mu=np.array([0, 0, 0, 1]),
        kappa=kappa,
    ).rvs(num_poses)

    include_identity = True
    if include_identity:
        position_deltas[0, :] = 0.0
        quaternion_deltas[0, :] = np.array([0, 0, 0, 1])

    pose_deltas = wp.array(
        np.hstack((position_deltas, quaternion_deltas)), dtype=wp.transform
    )
    c2f_schedule.append(pose_deltas)


def initialize(
    frame: Frame,
    object_mesh: Trimesh,
    object_idx=0,
    debug=False,
    seed=0,
):
    global pose_hypotheses, pixel_coordinates, corresponding_rgbd_per_pose_and_vertex, scores_per_pose_and_vertex, scores_per_pose
    spatial_means = object_mesh.vertices
    
    trimesh_mesh = object_mesh
    if not isinstance(trimesh_mesh.visual, trimesh.visual.color.ColorVisuals):
        vertex_colors = (
            np.array(trimesh_mesh.visual.to_color().vertex_colors)[..., :3] / 255.0
        )
    else:
        vertex_colors = (
            np.array(trimesh_mesh.visual.vertex_colors)[..., :3] / 255.0
        )
        
    rgb_means = vertex_colors * 255.0
    initial_object_pose_in_camera_frame = (condorgmm.Pose(frame.camera_pose).inv() @ condorgmm.Pose(frame.object_poses[object_idx]))

    transformed_points = initial_object_pose_in_camera_frame.apply(spatial_means)
    proj_pixel_coords = (transformed_points[:, :2] / transformed_points[:, 2:3]) * np.array([frame.intrinsics[0], frame.intrinsics[1]]) + np.array([frame.intrinsics[2], frame.intrinsics[3]])
    rounded_pixel_coordinates = np.floor(proj_pixel_coords).astype(np.int32)
    associated_rgb = frame.rgb[rounded_pixel_coordinates[:, 1], rounded_pixel_coordinates[:, 0]]
    associated_depth = frame.depth[rounded_pixel_coordinates[:, 1], rounded_pixel_coordinates[:, 0]]
    matching = np.abs(associated_depth - transformed_points[:, 2]) < 0.01

    # spatial_means = spatial_means[matching]
    rgb_means[matching,:] = associated_rgb[matching,:]
    
    num_poses = 20000

    gmm = warp_gmm.gmm_warp_from_numpy(
        spatial_means.astype(np.float32),
        rgb_means.astype(np.float32),
        object_posquats=initial_object_pose_in_camera_frame.posquat[None, ...].astype(np.float32),
        log_spatial_scales=np.log(0.0005 * np.ones((spatial_means.shape[0], 3), dtype=np.float32))
    )
    print(gmm.object_posquats.numpy()[0])


    num_vertices = gmm.spatial_means.shape[0]
    pose_hypotheses = wp.empty(num_poses, dtype=wp.transform)
    pixel_coordinates = wp.zeros((num_poses, num_vertices), dtype=wp.vec2i)
    corresponding_rgbd_per_pose_and_vertex = wp.empty(
        (num_poses, num_vertices), dtype=wp.vec4
    )
    scores_per_pose_and_vertex = wp.empty((num_poses, num_vertices), dtype=float)
    scores_per_pose = wp.zeros(num_poses, dtype=float)
    
    warp_gmm_state = warp_gmm.initialize_state(frame=frame)
    warp_gmm_state.gmm = gmm
    warp_gmm_state.gmm.object_posquats = wp.array(
        initial_object_pose_in_camera_frame.posquat[None, ...], dtype=wp.float32
    )

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
    gmm = warp_gmm_state.gmm
    for pose_deltas in c2f_schedule:
        inference_step(
            gmm.object_posquats,
            pose_deltas,
            gmm.spatial_means,
            gmm.rgb_means,
            frame.intrinsics[0],
            frame.intrinsics[1],
            frame.intrinsics[2],
            frame.intrinsics[3],
            frame_warp.rgb,
            frame_warp.depth,
            # These inputs are empty memory that will be filled by the kernels.
            pose_hypotheses,
            pixel_coordinates,
            corresponding_rgbd_per_pose_and_vertex,
            scores_per_pose_and_vertex,
            scores_per_pose,
        )

    object_pose = condorgmm.Pose(gmm.object_posquats.numpy()[0])
    transformed_points = object_pose.apply(gmm.spatial_means.numpy())
    proj_pixel_coords = (transformed_points[:, :2] / transformed_points[:, 2:3]) * np.array([frame.intrinsics[0], frame.intrinsics[1]]) + np.array([frame.intrinsics[2], frame.intrinsics[3]])
    rounded_pixel_coordinates = np.floor(proj_pixel_coords).astype(np.int32)
    valid = (rounded_pixel_coordinates[:, 0] >= 0) & (rounded_pixel_coordinates[:, 0] < frame.rgb.shape[1]) & (rounded_pixel_coordinates[:, 1] >= 0) & (rounded_pixel_coordinates[:, 1] < frame.rgb.shape[0])
    rounded_pixel_coordinates = rounded_pixel_coordinates * valid[:, None]
    associated_rgb = frame.rgb[rounded_pixel_coordinates[:, 1], rounded_pixel_coordinates[:, 0]]
    associated_depth = frame.depth[rounded_pixel_coordinates[:, 1], rounded_pixel_coordinates[:, 0]]
    matching = np.abs(associated_depth - transformed_points[:, 2]) < 0.005
    matching_and_valid = matching & valid
    rgb_means = gmm.rgb_means.numpy()
    rgb_means[matching_and_valid,:] = associated_rgb[matching_and_valid,:]        
    gmm.rgb_means = wp.array(rgb_means, dtype=wp.vec3)


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

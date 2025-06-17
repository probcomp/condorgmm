# import numpy as np
# from scipy.spatial.transform import Rotation as R
# import warp as wp
# import rerun as rr
# from pathlib import Path
# import condorgmm
# import cv2
# import trimesh

# vec6 = wp.types.vector(length=6, dtype=wp.float32)


# def rr_init(name="demo"):
#     global _blueprint_logged
#     _blueprint_logged = False
#     rr.init(name)
#     rr.connect("127.0.0.1:8812")


# def rr_set_time(t=0):
#     rr.set_time_sequence("step", t)


# def rr_log_rgb(rgb, channel="rgb"):
#     rr.log(channel, rr.Image(rgb[..., :3]))


# def rr_log_depth(depth, channel="depth"):
#     rr.log(channel, rr.DepthImage(depth * 1.0))


# def rr_log_mask(mask, channel="mask"):
#     rr.log(channel, rr.DepthImage(mask * 1.0))


# def rr_log_cloud(cloud, channel="cloud", colors=None):
#     if colors is None:
#         rr.log(channel, rr.Points3D(cloud.reshape(-1, 3)))
#     else:
#         rr.log(channel, rr.Points3D(cloud.reshape(-1, 3), colors=colors.reshape(-1, 3)))


# def rr_log_transform(transform, channel="pose", scale=0.1):
#     position = transform[:3]
#     origins = np.tile(position[None, ...], (3, 1))
#     colors = np.eye(3)
#     rotation_matrix = R.from_quat(transform[3:]).as_matrix()
#     rr.log(
#         channel,
#         rr.Arrows3D(origins=origins, vectors=rotation_matrix.T * scale, colors=colors),
#     )


# def rr_log_wp_transform(wp_transform: wp.transform, channel="pose", scale=0.1):
#     rr_log_transform(np.array(wp_transform), channel=channel, scale=scale)


# def rr_log_observed_rgb_and_d(observed_rgb, observed_d, fx, fy, cx, cy):
#     xyz = xyz_from_depth_image(observed_d, fx, fy, cx, cy)
#     condorgmm.rr_log_cloud(xyz, "scene/observed", colors=observed_rgb)
#     condorgmm.rr_log_depth(observed_d, "depth/observed")
#     condorgmm.rr_log_rgb(observed_rgb, "rgb/observed")


# def rr_log_prediction(vertices, colors, pose_estimate, gt_pose=None):
#     condorgmm.rr_log_cloud(vertices.numpy(), "model", colors=colors.numpy())
#     condorgmm.rr_log_cloud(
#         transform_points(np.array(pose_estimate), vertices.numpy()),
#         "scene/model",
#         colors=colors.numpy(),
#     )
#     condorgmm.rr_log_wp_transform(pose_estimate, "pose_estimate")
#     if gt_pose is not None:
#         condorgmm.rr_log_wp_transform(gt_pose, "gt_pose")


# def discretize_rgb(rgb):
#     lab = cv2.cvtColor((rgb / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB).astype(
#         np.float32
#     )
#     angle = np.rad2deg(np.arctan2(lab[..., 2], lab[..., 1]))
#     rounding = 30
#     angle_rounded = np.round((angle / rounding)) * rounding

#     radius = 25.0
#     new_lab = np.stack(
#         [
#             np.ones_like(lab[..., 0]) * 50.0,
#             np.cos(np.deg2rad(angle_rounded)) * radius,
#             np.sin(np.deg2rad(angle_rounded)) * radius,
#         ],
#         axis=-1,
#     ).astype(np.float32)
#     new_rgb = cv2.cvtColor(new_lab, cv2.COLOR_LAB2RGB)

#     norm = np.linalg.norm(lab[..., 1:], axis=-1)
#     black = lab[..., 0] < 10
#     white = np.logical_or(lab[..., 0] > 80, norm < 20.0) * ~black

#     new_rgb = new_rgb * ~black[..., None]
#     new_rgb = np.ones_like(rgb) * white[..., None] + new_rgb * ~white[..., None]
#     return new_rgb


# def discretize_rgbd(rgbd):
#     rgb = rgbd[..., :3]
#     depth = rgbd[..., 3]
#     return np.concatenate([discretize_rgb(rgb), depth[..., None]], axis=-1)


# def get_root_path() -> Path:
#     return Path(Path(condorgmm.__file__).parents[2])


# def xyz_from_depth_image(z, fx, fy, cx, cy):
#     #     v, u = np.mgrid[: z.shape[0], : z.shape[1]]
#     x = (u - cx) / fx
#     y = (v - cy) / fy
#     xyz = np.stack([x, y, np.ones_like(x)], dtype=z.dtype, axis=-1) * z[..., None]
#     return xyz


# def project_points(points, fx, fy, cx, cy):
#     #     x = points[..., 0] / points[..., 2]
#     y = points[..., 1] / points[..., 2]
#     u = x * fx + cx
#     v = y * fy + cy
#     return np.stack([v, u], axis=-1)


# def transform_to_posematrix(np_posquat):
#     pos = np_posquat[:3]
#     quat = np_posquat[3:]
#     romatrix = R.from_quat(quat).as_matrix()
#     return np.vstack(
#         [np.hstack([romatrix, pos.reshape(3, -1)]), np.array([0, 0, 0, 1])]
#     )


# def pose_matrix_to_transform(pose_matrix):
#     #     pos = pose_matrix[:3, 3]
#     quat = R.from_matrix(pose_matrix[:3, :3]).as_quat()
#     return np.concatenate([pos, quat])


# def transform_to_wp_transform(transform):
#     return wp.transform(transform[:3], wp.quat(transform[3:]))


# def pose_matrix_to_wp_transform(pose_matrix):
#     #     return transform_to_wp_transform(pose_matrix_to_transform(pose_matrix))


# def transform_points(transform, points):
#     return R.from_quat(transform[3:]).apply(points) + transform[:3]


# @wp.kernel
# def transform_points_kernel(
#     points: wp.array(dtype=wp.vec3),
#     wp_transform: wp.array(dtype=wp.float32),
#     out_points: wp.array(dtype=wp.vec3),
# ):
#     tid = wp.tid()
#     transform = wp.transform(
#         wp.vec3(wp_transform[0], wp_transform[1], wp_transform[2]),
#         wp.quat(wp_transform[3], wp_transform[4], wp_transform[5], wp_transform[6]),
#     )
#     out_points[tid] = wp.transform_point(transform, points[tid])


# def transform_points_wp(wp_transform, points):
#     #     point_count = points.shape[0]
#     out_points = wp.zeros(point_count, dtype=wp.vec3)
#     wp.launch(
#         kernel=transform_points_kernel,
#         dim=point_count,
#         inputs=[points, wp_transform, out_points],
#     )
#     return out_points


# @wp.kernel
# def inverse_transform_kernel(
#     wp_transform: wp.array(dtype=wp.float32),
#     output_transform_array: wp.array(dtype=wp.float32),
# ):
#     transform = wp.transform(
#         wp.vec3(wp_transform[0], wp_transform[1], wp_transform[2]),
#         wp.quat(wp_transform[3], wp_transform[4], wp_transform[5], wp_transform[6]),
#     )
#     inverse_transform = wp.transform_inverse(transform)
#     output_transform_array[0] = inverse_transform[0]
#     output_transform_array[1] = inverse_transform[1]
#     output_transform_array[2] = inverse_transform[2]
#     output_transform_array[3] = inverse_transform[3]
#     output_transform_array[4] = inverse_transform[4]
#     output_transform_array[5] = inverse_transform[5]
#     output_transform_array[6] = inverse_transform[6]


# def inverse_transform(wp_transform):
#     #     output_transform_array = wp.zeros(7, dtype=wp.float32)
#     wp.launch(
#         kernel=inverse_transform_kernel,
#         dim=1,
#         inputs=[wp_transform, output_transform_array],
#     )
#     return output_transform_array


# def vertices_faces_colors_from_trimesh(trimesh_mesh):
#     vertices = np.array(trimesh_mesh.vertices)
#     faces = np.array(trimesh_mesh.faces)
#     if not isinstance(trimesh_mesh.visual, trimesh.visual.color.ColorVisuals):
#         vertex_colors = (
#             np.array(trimesh_mesh.visual.to_color().vertex_colors)[..., :3] / 255.0
#         )
#     else:
#         vertex_colors = np.array(trimesh_mesh.visual.vertex_colors)[..., :3] / 255.0
#     return (vertices, faces, vertex_colors)

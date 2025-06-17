# import warp as wp


# @wp.kernel
# def warp_gmm_sample_kernel(
#     posequat: wp.array(dtype=wp.float32),
#     camera_posequat: wp.array(dtype=wp.float32),
#     spatial_means: wp.array(dtype=wp.vec3),
#     rgb_means: wp.array(dtype=wp.vec3),
#     spatial_scales: wp.array(dtype=wp.vec3),
#     rgb_scales: wp.array(dtype=wp.vec3),
#     quaternions_imaginary: wp.array(dtype=wp.vec3),
#     quaternions_real: wp.array(dtype=wp.float32),
#     sampled_points: wp.array(dtype=wp.vec3),
#     sampled_colors: wp.array(dtype=wp.vec3),
# ):
#     # Get thread ID
#     idx = wp.tid()

#     # Initialize random state
#     state = wp.rand_init(wp.int32(idx))

#     # Sample Gaussian component uniformly
#     component_idx = wp.randi(state, 0, spatial_means.shape[0])

#     # Get component parameters
#     mean_xyz = spatial_means[component_idx]
#     mean_rgb = rgb_means[component_idx]
#     scale_xyz = spatial_scales[component_idx]
#     scale_rgb = rgb_scales[component_idx]

#     # Construct transform
#     pose_transform = wp.transform(
#         wp.vec3(posequat[0], posequat[1], posequat[2]),
#         wp.quat(posequat[3], posequat[4], posequat[5], posequat[6]),
#     )

#     point_transform = wp.transform(
#         wp.vec3(mean_xyz[0], mean_xyz[1], mean_xyz[2]),
#         wp.quat(
#             quaternions_real[component_idx],
#             quaternions_imaginary[component_idx][0],
#             quaternions_imaginary[component_idx][1],
#             quaternions_imaginary[component_idx][2],
#         ),
#     )

#     transform = wp.transform_multiply(pose_transform, point_transform)
#     rot_matrix = wp.quat_to_matrix(wp.transform_get_rotation(transform))

#     # Sample position from anisotropic Gaussian
#     noise_xyz = wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))
#     scaled_noise = wp.vec3(
#         noise_xyz[0] * wp.exp(scale_xyz[0]),
#         noise_xyz[1] * wp.exp(scale_xyz[1]),
#         noise_xyz[2] * wp.exp(scale_xyz[2]),
#     )

#     rotated_noise = rot_matrix * scaled_noise
#     sampled_pos = wp.transform_get_translation(transform) + rotated_noise

#     # Sample color from isotropic Gaussian
#     noise_rgb = wp.vec3(
#         wp.randn(state) * wp.exp(scale_rgb[0]),
#         wp.randn(state) * wp.exp(scale_rgb[1]),
#         wp.randn(state) * wp.exp(scale_rgb[2]),
#     )
#     sampled_color = mean_rgb + noise_rgb

#     # Store results
#     sampled_points[idx] = sampled_pos
#     sampled_colors[idx] = sampled_color


# def warp_gmm_sample(
#     posequat,
#     camera_posequat,
#     spatial_means,
#     rgb_means,
#     spatial_scales,
#     rgb_scales,
#     quaternions_imaginary,
#     quaternions_real,
#     num_samples,
# ):
#     sampled_points = wp.zeros(num_samples, dtype=wp.vec3)
#     sampled_colors = wp.zeros(num_samples, dtype=wp.vec3)
#     wp.launch(
#         kernel=warp_gmm_sample_kernel,
#         dim=num_samples,
#         inputs=[
#             posequat,
#             camera_posequat,
#             spatial_means,
#             rgb_means,
#             spatial_scales,
#             rgb_scales,
#             quaternions_imaginary,
#             quaternions_real,
#             sampled_points,
#             sampled_colors,
#         ],
#     )
#     return sampled_points, sampled_colors

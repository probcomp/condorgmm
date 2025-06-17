import warp as wp
import numpy as np
import condorgmm
from .state import State


snippet = adj_snippet = @wp.func_native(snippet, adj_snippet)
def warp_gmm_native_log_add_exp(
    log_value_to_add_in: float,
    unraveled_index: int,
    log_score_image: wp.array(dtype=wp.float32, ndim=2),
): ...


@wp.kernel
def warp_gmm_forward_kernel(
    object_posquats: wp.array(ndim=2, dtype=wp.float32),
    camera_posquat: wp.array(dtype=wp.float32),
    spatial_means: wp.array(dtype=wp.vec3),
    rgb_means: wp.array(dtype=wp.vec3),
    log_spatial_scales: wp.array(dtype=wp.vec3),
    log_rgb_scales: wp.array(dtype=wp.vec3),
    quaternions_imaginary: wp.array(dtype=wp.vec3),
    quaternions_real: wp.array(dtype=wp.float32),
    log_mixture_weights: wp.array(dtype=wp.float32),
    assignments: wp.array(dtype=wp.int32),
    outlier_probability: float,
    outlier_volume: float,
    mask: wp.array(dtype=wp.bool, ndim=2),
    gaussian_mask: wp.array(dtype=wp.bool, ndim=1),
    observed_rgb_image: wp.array(dtype=wp.vec3, ndim=2),
    observed_depth_image: wp.array(dtype=wp.float32, ndim=2),
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    window_half_width: int,
    log_score_image: wp.array(dtype=wp.float32, ndim=2),
    log_score_per_gaussian_per_pixel: wp.array(dtype=wp.float32, ndim=3),
    write_log_score_per_gaussian_per_pixel: bool,
):
    gaussian_index, offset_i, offset_j = wp.tid()

    if not gaussian_mask[gaussian_index]:
        return

    xyz = spatial_means[gaussian_index]

    object_index = assignments[gaussian_index]

    pose_transform = wp.transform(
        wp.vec3(
            object_posquats[object_index, 0],
            object_posquats[object_index, 1],
            object_posquats[object_index, 2],
        ),
        safe_normalize(
            wp.quat(
                object_posquats[object_index, 3],
                object_posquats[object_index, 4],
                object_posquats[object_index, 5],
                object_posquats[object_index, 6],
            )
        ),
    )

    camera_transform = wp.transform(
        wp.vec3(camera_posquat[0], camera_posquat[1], camera_posquat[2]),
        safe_normalize(
            wp.quat(
                camera_posquat[3],
                camera_posquat[4],
                camera_posquat[5],
                camera_posquat[6],
            )
        ),
    )
    quaternion = safe_normalize(
        wp.quat(quaternions_imaginary[gaussian_index], quaternions_real[gaussian_index])
    )

    point_transform = wp.transform(wp.vec3(xyz[0], xyz[1], xyz[2]), quaternion)

    view_frame_transform = wp.transform_multiply(
        wp.transform_inverse(camera_transform), pose_transform
    )
    point_transform_in_view_frame = wp.transform_multiply(
        view_frame_transform, point_transform
    )

    gaussian_xyz = wp.transform_get_translation(point_transform_in_view_frame)
    gaussian_quat = wp.transform_get_rotation(point_transform_in_view_frame)

    if gaussian_xyz[2] <= 1e-10:
        return

    discrete_center_pixel = wp.vec2i(
        wp.int32(wp.floor(fy * gaussian_xyz[1] / (gaussian_xyz[2]) + cy)),
        wp.int32(wp.floor(fx * gaussian_xyz[0] / (gaussian_xyz[2]) + cx)),
    )

    pixel_i = discrete_center_pixel[0] + offset_i - window_half_width
    pixel_j = discrete_center_pixel[1] + offset_j - window_half_width
    if pixel_i < 0 or pixel_i >= height or pixel_j < 0 or pixel_j >= width:
        return

    if not mask[pixel_i, pixel_j]:
        return

    observed_rgb_pixel = observed_rgb_image[pixel_i, pixel_j]
    observed_depth_pixel = observed_depth_image[pixel_i, pixel_j]

    # Add small epsilon to avoid division by zero
    observed_xyz_pixel = wp.vec3(
        (wp.float32(pixel_j) + 0.5 - cx) / fx * (observed_depth_pixel),
        (wp.float32(pixel_i) + 0.5 - cy) / fy * (observed_depth_pixel),
        observed_depth_pixel,
    )

    difference_xyz = observed_xyz_pixel - gaussian_xyz
    difference_rgb = wp.vec3(observed_rgb_pixel) - wp.vec3(rgb_means[gaussian_index])

    total_log_score = wp.float32(0.0)
    total_log_score += wp.log(wp.max(1.0 - outlier_probability, 1e-20))

    total_log_score += log_mixture_weights[gaussian_index]

    gaussian_scales_xyz = wp.vec3(
        wp.exp(log_spatial_scales[gaussian_index][0]) + 1e-20,
        wp.exp(log_spatial_scales[gaussian_index][1]) + 1e-20,
        wp.exp(log_spatial_scales[gaussian_index][2]) + 1e-20,
    )

    gaussian_quat = safe_normalize(gaussian_quat)
    rot_matrix = wp.quat_to_matrix(gaussian_quat)

    cov = (
        rot_matrix
        * wp.diag(gaussian_scales_xyz)
        * wp.diag(gaussian_scales_xyz)
        * wp.transpose(rot_matrix)
    )  # + wp.diag(diagonal_epsilon)

    det = wp.determinant(cov)
    log_det_cov = wp.log(wp.max(det, 1e-20))

    cov_inv = wp.inverse(cov)

    logpdf_xyz = -0.5 * (
        3.0 * wp.log(2.0 * wp.pi)
        + log_det_cov
        + wp.dot(difference_xyz, cov_inv * difference_xyz)
    )
    total_log_score += logpdf_xyz

    gaussian_scales_rgb = wp.vec3(
        wp.exp(log_rgb_scales[gaussian_index][0]) + 1e-20,
        wp.exp(log_rgb_scales[gaussian_index][1]) + 1e-20,
        wp.exp(log_rgb_scales[gaussian_index][2]) + 1e-20,
    )
    for i in range(wp.static(3)):
        scale = gaussian_scales_rgb[i]
        total_log_score += -0.5 * wp.log(2.0 * wp.pi * wp.pow(scale, 2.0) + 1e-20)
        total_log_score += -0.5 * (
            wp.pow(difference_rgb[i], 2.0) / (wp.pow(scale, 2.0) + 1e-20)
        )

    warp_gmm_native_log_add_exp(
        total_log_score, pixel_i * width + pixel_j, log_score_image
    )

    if write_log_score_per_gaussian_per_pixel:
        log_score_per_gaussian_per_pixel[gaussian_index, offset_i, offset_j] = (
            total_log_score
        )


@wp.func
def safe_normalize(
    quaternion: wp.quat,
):
    return quaternion / wp.sqrt(wp.dot(quaternion, quaternion) + 1e-20)


# def normalize_quaternions(
#     object_posquats: wp.array(ndim=2, dtype=wp.float32),
#     camera_posquat: wp.array(dtype=wp.float32),
#     quaternions_imaginary: wp.array(dtype=wp.vec3),
#     quaternions_real: wp.array(dtype=wp.float32),
# ):
#     index = wp.tid()

#     if index == 0:
#         quaternion = safe_normalize(
#             wp.quat(quaternions_imaginary[i], quaternions_real[i])
#         )

#     for i in range(quaternions_imaginary.shape[0]):
#         quaternion = safe_normalize(
#             wp.quat(quaternions_imaginary[i], quaternions_real[i])
#         )
#         quaternions_imaginary[i] = quaternion[0], quaternion[1], quaternion[2]


@wp.kernel
def warp_gmm_EM_step_kernel(
    object_posquats: wp.array(ndim=2, dtype=wp.float32),
    camera_posquat: wp.array(dtype=wp.float32),
    spatial_means: wp.array(dtype=wp.vec3),
    rgb_means: wp.array(dtype=wp.vec3),
    log_spatial_scales: wp.array(dtype=wp.vec3),
    log_rgb_scales: wp.array(dtype=wp.vec3),
    quaternions_imaginary: wp.array(dtype=wp.vec3),
    quaternions_real: wp.array(dtype=wp.float32),
    log_mixture_weights: wp.array(dtype=wp.float32),
    assignments: wp.array(dtype=wp.int32),
    outlier_probability: float,
    outlier_volume: float,
    mask: wp.array(dtype=wp.bool, ndim=2),
    observed_rgb_image: wp.array(dtype=wp.vec3, ndim=2),
    observed_depth_image: wp.array(dtype=wp.float32, ndim=2),
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    window_half_width: int,
    log_score_image: wp.array(dtype=wp.float32, ndim=2),
    log_score_per_gaussian_per_pixel: wp.array(dtype=wp.float32, ndim=3),
):
    gaussian_index, offset_i, offset_j = wp.tid()

    xyz = spatial_means[gaussian_index]

    object_index = assignments[gaussian_index]

    pose_transform = wp.transform(
        wp.vec3(
            object_posquats[object_index, 0],
            object_posquats[object_index, 1],
            object_posquats[object_index, 2],
        ),
        safe_normalize(
            wp.quat(
                object_posquats[object_index, 3],
                object_posquats[object_index, 4],
                object_posquats[object_index, 5],
                object_posquats[object_index, 6],
            )
        ),
    )

    camera_transform = wp.transform(
        wp.vec3(camera_posquat[0], camera_posquat[1], camera_posquat[2]),
        safe_normalize(
            wp.quat(
                camera_posquat[3],
                camera_posquat[4],
                camera_posquat[5],
                camera_posquat[6],
            )
        ),
    )
    quaternion = safe_normalize(
        wp.quat(quaternions_imaginary[gaussian_index], quaternions_real[gaussian_index])
    )
    quaternions_imaginary[gaussian_index] = wp.vec3(
        quaternion[0], quaternion[1], quaternion[2]
    )
    quaternions_real[gaussian_index] = quaternion[3]

    point_transform = wp.transform(wp.vec3(xyz[0], xyz[1], xyz[2]), quaternion)

    view_frame_transform = wp.transform_multiply(
        wp.transform_inverse(camera_transform), pose_transform
    )
    point_transform_in_view_frame = wp.transform_multiply(
        view_frame_transform, point_transform
    )
    gaussian_xyz = wp.transform_get_translation(point_transform_in_view_frame)
    # gaussian_quat = wp.transform_get_rotation(point_transform_in_view_frame)

    if gaussian_xyz[2] < 0.0:
        return

    discrete_center_pixel = wp.vec2i(
        wp.int32(wp.floor(fy * gaussian_xyz[1] / (gaussian_xyz[2]) + cy)),
        wp.int32(wp.floor(fx * gaussian_xyz[0] / (gaussian_xyz[2]) + cx)),
    )

    new_spatial_means = wp.vec3(0.0, 0.0, 0.0)
    new_rgb_means = wp.vec3(0.0, 0.0, 0.0)
    new_mixture_weight = wp.float32(0.0)

    for offset_i in range(2 * window_half_width + 1):
        for offset_j in range(2 * window_half_width + 1):
            pixel_i = discrete_center_pixel[0] + offset_i - window_half_width
            pixel_j = discrete_center_pixel[1] + offset_j - window_half_width
            if pixel_i < 0 or pixel_i >= height or pixel_j < 0 or pixel_j >= width:
                continue

            if not mask[pixel_i, pixel_j]:
                continue

            observed_rgb_pixel = observed_rgb_image[pixel_i, pixel_j]
            observed_depth_pixel = observed_depth_image[pixel_i, pixel_j]
            observed_xyz_pixel = wp.vec3(
                (wp.float32(pixel_j) + 0.5 - cx) / fx * observed_depth_pixel,
                (wp.float32(pixel_i) + 0.5 - cy) / fy * observed_depth_pixel,
                observed_depth_pixel,
            )

            ratio = wp.exp(
                log_score_per_gaussian_per_pixel[gaussian_index, offset_i, offset_j]
                - log_score_image[pixel_i, pixel_j]
            )

            # Update spatial mean
            new_spatial_means += ratio * observed_xyz_pixel
            # Update RGB mean
            new_rgb_means += ratio * observed_rgb_pixel

            new_mixture_weight += ratio

    new_spatial_means = new_spatial_means / (new_mixture_weight + 1e-10)
    new_rgb_means = new_rgb_means / (new_mixture_weight + 1e-10)

    new_spatial_covariances = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    new_rgb_variances = wp.vec3(0.0, 0.0, 0.0)

    for offset_i in range(2 * window_half_width + 1):
        for offset_j in range(2 * window_half_width + 1):
            pixel_i = discrete_center_pixel[0] + offset_i - window_half_width
            pixel_j = discrete_center_pixel[1] + offset_j - window_half_width
            if pixel_i < 0 or pixel_i >= height or pixel_j < 0 or pixel_j >= width:
                continue

            if not mask[pixel_i, pixel_j]:
                continue

            observed_rgb_pixel = observed_rgb_image[pixel_i, pixel_j]
            observed_depth_pixel = observed_depth_image[pixel_i, pixel_j]
            observed_xyz_pixel = wp.vec3(
                (wp.float32(pixel_j) + 0.5 - cx) / fx * observed_depth_pixel,
                (wp.float32(pixel_i) + 0.5 - cy) / fy * observed_depth_pixel,
                observed_depth_pixel,
            )

            ratio = wp.exp(
                log_score_per_gaussian_per_pixel[gaussian_index, offset_i, offset_j]
                - log_score_image[pixel_i, pixel_j]
            )

            diff = observed_xyz_pixel - new_spatial_means
            new_spatial_covariances += ratio * wp.outer(diff, diff)
            # Update RGB variances
            rgb_diff = observed_rgb_pixel - new_rgb_means
            new_rgb_variances += ratio * wp.vec3(
                rgb_diff[0] * rgb_diff[0],
                rgb_diff[1] * rgb_diff[1],
                rgb_diff[2] * rgb_diff[2],
            )

    new_spatial_covariances = new_spatial_covariances / (new_mixture_weight + 1e-10)
    new_rgb_variances = new_rgb_variances / (new_mixture_weight + 1e-10)

    new_mixture_weight = new_mixture_weight / wp.float32(spatial_means.shape[0])

    Q = wp.mat33()
    d = wp.vec3()

    wp.eig3(new_spatial_covariances, Q, d)
    quaternion = wp.quat_from_matrix(Q)

    new_point_transform = wp.transform(new_spatial_means, quaternion)

    new_point_transform_in_object_frame = wp.transform_multiply(
        wp.transform_inverse(view_frame_transform), new_point_transform
    )

    spatial_means[gaussian_index] = wp.transform_get_translation(
        new_point_transform_in_object_frame
    )
    new_quaternion = wp.transform_get_rotation(new_point_transform_in_object_frame)
    quaternions_imaginary[gaussian_index] = wp.vec3(
        new_quaternion[0], new_quaternion[1], new_quaternion[2]
    )
    quaternions_real[gaussian_index] = new_quaternion[3]

    rgb_means[gaussian_index] = new_rgb_means

    log_spatial_scales[gaussian_index] = wp.vec3(
        wp.log(wp.sqrt(d[0] + 1e-6) + 1e-6),
        wp.log(wp.sqrt(d[1] + 1e-6) + 1e-6),
        wp.log(wp.sqrt(d[2] + 1e-6) + 1e-6),
    )
    log_rgb_scales[gaussian_index] = wp.vec3(
        wp.log(wp.sqrt(new_rgb_variances[0] + 1e-6) + 1e-6),
        wp.log(wp.sqrt(new_rgb_variances[1] + 1e-6) + 1e-6),
        wp.log(wp.sqrt(new_rgb_variances[2] + 1e-6) + 1e-6),
    )

    # log_mixture_weights[gaussian_index] = wp.log(new_mixture_weight)


##### Interface


def warp_gmm_forward(
    frame: condorgmm.Frame,
    state: State,
    log_score_per_gaussian_per_pixel=None,
    write_log_score_per_gaussian_per_pixel=False,
):
    height, width = frame.rgb.shape[:2]
    fx, fy, cx, cy = frame.intrinsics
    hyperparams = state.hyperparams
    gmm = state.gmm
    
    if state.gaussian_mask is None:
        gaussian_mask = wp.ones(gmm.spatial_means.shape[0], dtype=wp.bool)
    else:
        gaussian_mask = state.gaussian_mask

    state.log_score_image.fill_(
        np.log(hyperparams.outlier_probability * 1.0 / hyperparams.outlier_volume)
    )
    wp.launch(
        kernel=warp_gmm_forward_kernel,
        dim=(
            gmm.spatial_means.shape[0],
            (2 * hyperparams.window_half_width + 1),
            (2 * hyperparams.window_half_width + 1),
        ),
        inputs=[
            gmm.object_posquats,
            gmm.camera_posquat,
            gmm.spatial_means,
            gmm.rgb_means,
            gmm.log_spatial_scales,
            gmm.log_rgb_scales,
            gmm.quaternions_imaginary,
            gmm.quaternions_real,
            gmm.log_mixture_weights,
            gmm.assignments,
            hyperparams.outlier_probability,
            hyperparams.outlier_volume,
            state.mask,
            gaussian_mask,
            frame.rgb,
            frame.depth,
            height,
            width,
            fx,
            fy,
            cx,
            cy,
            hyperparams.window_half_width,
            state.log_score_image,
            log_score_per_gaussian_per_pixel,
            write_log_score_per_gaussian_per_pixel,
        ],
    )


def warp_gmm_EM_step(
    frame: condorgmm.Frame,
    state: State,
):
    hyperparams = state.hyperparams
    gmm = state.gmm
    window_half_width = hyperparams.window_half_width

    height, width = frame.height, frame.width
    log_score_per_gaussian_per_pixel = wp.zeros(
        (
            gmm.spatial_means.shape[0],
            2 * window_half_width + 1,
            2 * window_half_width + 1,
        ),
        dtype=wp.float32,
    )
    log_score_per_gaussian_per_pixel.fill_(-np.inf)

    warp_gmm_forward(frame, state, log_score_per_gaussian_per_pixel, True)

    fx, fy, cx, cy = frame.intrinsics

    wp.launch(
        kernel=warp_gmm_EM_step_kernel,
        dim=(gmm.spatial_means.shape[0],),
        inputs=[
            gmm.object_posquats,
            gmm.camera_posquat,
            gmm.spatial_means,
            gmm.rgb_means,
            gmm.log_spatial_scales,
            gmm.log_rgb_scales,
            gmm.quaternions_imaginary,
            gmm.quaternions_real,
            gmm.log_mixture_weights,
            gmm.assignments,
            hyperparams.outlier_probability,
            hyperparams.outlier_volume,
            state.mask,
            frame.rgb,
            frame.depth,
            height,
            width,
            fx,
            fy,
            cx,
            cy,
            window_half_width,
            state.log_score_image,
            log_score_per_gaussian_per_pixel,
        ],
    )

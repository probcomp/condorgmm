import warp as wp
import numpy as np

@wp.func
def gaussian_likelhood(x: float, std_dev: float):
    log_coefficient = -0.5 * wp.log(2.0 * wp.pi * wp.pow(std_dev, 2.0))
    log_exponent = -0.5 * wp.pow((x / std_dev), 2.0)
    return log_coefficient + log_exponent


@wp.func
def pixel_likelihood(
    pixel_rgb: wp.vec3,
    pixel_depth: wp.float32,
    spatial_mean: wp.vec3,
    rgb_mean: wp.vec3,
    is_outlier: wp.bool,
):
    depth_score = gaussian_likelhood(pixel_depth - spatial_mean[2], 0.005)
    r_score = gaussian_likelhood(pixel_rgb[0] - rgb_mean[0], 10.1)
    g_score = gaussian_likelhood(pixel_rgb[1] - rgb_mean[1], 10.1)
    b_score = gaussian_likelhood(pixel_rgb[2] - rgb_mean[2], 10.1)
    rgb_score = r_score + g_score + b_score

    outlier_prob = 0.3
    if is_outlier:
        return wp.log(outlier_prob) + wp.log(1.0/10000.0)
    else:
        return rgb_score + depth_score + wp.log(1.0 - outlier_prob)


@wp.kernel
def apply_pose_deltas(
    pose_center_posquats: wp.array(ndim=2, dtype=wp.float32),
    pose_deltas: wp.array(dtype=wp.transform),
    pose_hypotheses: wp.array(dtype=wp.transform),
):
    pose_index = wp.tid()
    
    pose_center = wp.transform(wp.vec3(pose_center_posquats[0, 0], pose_center_posquats[0, 1], pose_center_posquats[0, 2]),
                               wp.quat(pose_center_posquats[0, 3], pose_center_posquats[0, 4], pose_center_posquats[0, 5], pose_center_posquats[0, 6]))
    pose_hypotheses[pose_index] = wp.transform_multiply(
        pose_center, pose_deltas[pose_index]
    )


@wp.kernel
def score_pose_and_vertex(
    observed_rgb_image: wp.array(dtype=wp.vec3, ndim=2),
    observed_depth_image: wp.array(dtype=wp.float32, ndim=2),
    spatial_means: wp.array(dtype=wp.vec3),
    rgb_means: wp.array(dtype=wp.vec3),
    pose_hypotheses: wp.array(dtype=wp.transform),
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pixel_coordinates: wp.array(dtype=wp.vec2i, ndim=2),
    corresponding_rgbd_per_pose_and_vertex: wp.array(dtype=wp.vec4, ndim=2),
    scores_per_pose_and_vertex: wp.array(dtype=float, ndim=2),
):
    # Each thread will process a pair of a pose hypothesis and a vertex.
    (pose_index, vertex_index) = wp.tid()

    # Transform the assigned vertex by the the assigned pose hypothesis.
    transformed_vertex = wp.transform_point(
        pose_hypotheses[pose_index], spatial_means[vertex_index]
    )

    # Project the transformed vertex to the image plane.
    pixel_raw = wp.vec2i(
        wp.int32(fy * transformed_vertex[1] / transformed_vertex[2] + cy),
        wp.int32(fx * transformed_vertex[0] / transformed_vertex[2] + cx),
    )

    # Check whether the projection is valid.
    # (1) Is it within the dimensions of the image.
    # (2) Is it in front of the camera.
    height = observed_rgb_image.shape[0]
    width = observed_rgb_image.shape[1]
    valid = (
        pixel_raw[0] >= 0
        and pixel_raw[0] < height
        and pixel_raw[1] >= 0
        and pixel_raw[1] < width
        and transformed_vertex[2] > 0
    )

    # Clip the pixel coordinates to the image dimensions to avoid out-of-bounds access.
    pixel = wp.vec2i(
        min(max(pixel_raw[0], 0), height - 1),
        min(max(pixel_raw[1], 0), width - 1),
    )

    # Get the observed RGBD pixel at the projected pixel coordinates.
    observed_rgb_pixel = observed_rgb_image[pixel[0], pixel[1]]
    observed_depth_pixel = observed_depth_image[pixel[0], pixel[1]]
    # If it's invalid, then set the pixel values to be invalid.
    if not valid:
        observed_rgb_pixel = wp.vec3(-1.0, -1.0, -1.0)
        observed_depth_pixel = -1.0

    # Now grid over the outlier status of the pixel and calculate the scores.
    scores = wp.vector(0.0, 0.0)
    sweep_over_is_outlier = wp.vector(wp.bool(True), wp.bool(False))

    for i in range(wp.static(2)):
        scores[i] = pixel_likelihood(
            observed_rgb_pixel,
            observed_depth_pixel,
            transformed_vertex,
            rgb_means[vertex_index],
            sweep_over_is_outlier[i],
        )

    # Record the pixel likelihood score.
    scores_per_pose_and_vertex[pose_index, vertex_index] = scores[
        int(wp.argmax(scores))
    ]

    #### All the following code is outputs that are used for debugging. ###

    # Save the pixel coordinates for the pose and vertex.
    pixel_coordinates[pose_index, vertex_index] = pixel_raw
    # This is for debugging
    corresponding_rgbd_per_pose_and_vertex[pose_index, vertex_index] = (
        wp.vec4(observed_rgb_pixel[0], observed_rgb_pixel[1], observed_rgb_pixel[2], observed_depth_pixel)
    )


@wp.kernel
def accumulate_scores(
    scores_per_pose_and_vertex: wp.array(dtype=float, ndim=2),
    scores_per_pose: wp.array(dtype=float),
):
    pose_index = wp.tid()
    accumulated_score = float(0.0)
    for vertex_index in range(scores_per_pose_and_vertex.shape[1]):
        accumulated_score = (
            accumulated_score + scores_per_pose_and_vertex[pose_index, vertex_index]
        )
    scores_per_pose[pose_index] = accumulated_score


@wp.kernel
def select_best_pose(
    pose_hypotheses: wp.array(dtype=wp.transform),
    scores_per_pose: wp.array(dtype=float),
    object_posquats: wp.array(ndim=2, dtype=wp.float32),
):
    best_index = int(0)
    best_value = float(-np.inf)
    for i in range(pose_hypotheses.shape[0]):
        if scores_per_pose[i] > best_value:
            best_value = scores_per_pose[i]
            best_index = i
            
    best_pose = pose_hypotheses[best_index]
    best_translation = wp.transform_get_translation(best_pose)
    best_rotation = wp.transform_get_rotation(best_pose)
    object_posquats[0, 0] = best_translation[0]
    object_posquats[0, 1] = best_translation[1]
    object_posquats[0, 2] = best_translation[2]
    object_posquats[0, 3] = best_rotation[0]
    object_posquats[0, 4] = best_rotation[1]
    object_posquats[0, 5] = best_rotation[2]
    object_posquats[0, 6] = best_rotation[3]

def inference_step(
    object_posquats: wp.array(ndim=2, dtype=wp.float32),
    pose_deltas: wp.array(dtype=wp.transform),
    spatial_means: wp.array(dtype=wp.vec3),
    rgb_means: wp.array(dtype=wp.vec3),
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    observed_rgb_image: wp.array(dtype=wp.vec3, ndim=2),
    observed_depth_image: wp.array(dtype=wp.float32, ndim=2),
    # These inputs are empty memory that will be filled by the kernels.
    pose_hypotheses: wp.array(dtype=wp.transform),
    pixel_coordinates: wp.array(dtype=wp.vec2i, ndim=2),
    corresponding_rgbd_per_pose_and_vertex: wp.array(dtype=wp.vec4, ndim=2),
    scores_per_pose_and_vertex: wp.array(dtype=float, ndim=2),
    scores_per_pose: wp.array(dtype=float),
):
    num_poses = pose_deltas.shape[0]
    num_points = spatial_means.shape[0]

    wp.launch(
        kernel=apply_pose_deltas,
        dim=num_poses,
        inputs=[object_posquats, pose_deltas, pose_hypotheses],
    )
    wp.launch(
        kernel=score_pose_and_vertex,
        dim=(num_poses, num_points),
        inputs=[
            observed_rgb_image,
            observed_depth_image,
            spatial_means,
            rgb_means,
            pose_hypotheses,
            fx,
            fy,
            cx,
            cy,
            pixel_coordinates,
            corresponding_rgbd_per_pose_and_vertex,
            scores_per_pose_and_vertex,
        ],
    )
    wp.launch(
        kernel=accumulate_scores,
        dim=(num_poses),
        inputs=[scores_per_pose_and_vertex, scores_per_pose],
    )
    wp.launch(
        kernel=select_best_pose,
        dim=(1),
        inputs=[pose_hypotheses, scores_per_pose, object_posquats],
    )
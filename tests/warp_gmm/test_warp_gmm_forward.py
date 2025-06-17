import numpy as np
import warp as wp
import condorgmm.warp_gmm as warp_gmm
from tqdm import tqdm
import scipy.stats
from scipy.spatial.transform import Rotation as Rot
import condorgmm


def test_warp_gmm_matches_hand_implementation():
    height, width = 480, 640
    number_of_gaussians = 20

    xyzs = wp.array(
        (
            (np.random.rand(number_of_gaussians, 3).astype(np.float32) - 0.5) * 0.3
            + np.array([0.0, 0.0, 1.0])[None, ...]
        ),
        dtype=wp.vec3,
    )
    rgbs = wp.array(
        (np.random.rand(number_of_gaussians, 3).astype(np.float32)), dtype=wp.vec3
    )
    log_scales_xyz = wp.array(
        np.log(np.random.rand(number_of_gaussians, 3).astype(np.float32)), dtype=wp.vec3
    )
    log_scales_rgb = wp.array(
        np.log(np.random.rand(number_of_gaussians, 3).astype(np.float32)),
        dtype=wp.vec3,
    )
    quaternions_imaginary = wp.array(
        (np.random.rand(number_of_gaussians, 3).astype(np.float32)),
        dtype=wp.vec3,
    )
    quaternions_real = wp.array(
        (np.random.rand(number_of_gaussians).astype(np.float32)),
        dtype=wp.float32,
    )

    fx, fy, cx, cy = 1000.0, 1000.0, width / 2.0, height / 2.0

    observed_rgb_image = wp.array(
        np.random.rand(height, width, 3).astype(np.float32), dtype=wp.vec3
    )
    observed_depth_image = wp.array(
        np.random.rand(height, width).astype(np.float32), dtype=wp.float32
    )

    frame_warp = condorgmm.Frame(
        rgb=observed_rgb_image,
        depth=observed_depth_image,
        intrinsics=(fx, fy, cx, cy),
    )

    log_mixture_weights = wp.zeros(
        number_of_gaussians, dtype=wp.float32, requires_grad=True
    )

    gmm = warp_gmm.gmm_warp_constructor(
        xyzs,
        rgbs,
        log_scales_xyz,
        log_scales_rgb,
        quaternions_imaginary,
        quaternions_real,
        log_mixture_weights,
    )

    warp_gmm_state = warp_gmm.initialize_state(gmm, height=height, width=width)
    warp_gmm.warp_gmm_forward(
        frame_warp,
        warp_gmm_state,
    )
    hyperparams = warp_gmm_state.hyperparams

    log_score_image_np = warp_gmm_state.log_score_image.numpy().view(np.float32)

    gaussian_xyzs = xyzs.numpy()
    gaussian_rgbs = rgbs.numpy()

    scales_rgb = np.exp(log_scales_rgb.numpy())
    scales_xyz = np.exp(log_scales_xyz.numpy())

    covariances_np = np.zeros((number_of_gaussians, 6, 6))
    quaternions_np = np.hstack(
        [quaternions_imaginary.numpy(), quaternions_real.numpy().reshape(-1, 1)]
    )  # quaternions.numpy()

    # check rotation matrices match
    for i in range(number_of_gaussians):
        rotation_matrix = Rot.from_quat(quaternions_np[i]).as_matrix()
        warp_rotation_matrix = np.array(
            wp.quat_to_matrix(wp.normalize(wp.quat(quaternions_np[i])))
        ).reshape(3, 3)
        assert np.isclose(np.linalg.det(rotation_matrix), 1.0)
        assert np.isclose(np.linalg.det(warp_rotation_matrix), 1.0)
        assert np.allclose(rotation_matrix, warp_rotation_matrix, atol=1e-5)

    rotation_matrices = Rot.from_quat(quaternions_np).as_matrix()
    covariances_xyz = np.zeros((number_of_gaussians, 3, 3))
    covariances_xyz[:, np.arange(3), np.arange(3)] = scales_xyz**2
    covariances_xyz = (
        rotation_matrices @ covariances_xyz @ np.swapaxes(rotation_matrices, 1, 2)
    )

    covariances_np[:, :3, :3] = covariances_xyz
    covariances_np[:, np.arange(3, 6), np.arange(3, 6)] = scales_rgb**2

    rgb = observed_rgb_image.numpy()
    depth = observed_depth_image.numpy()

    log_score_image_reconstruction = np.log(
        np.ones((height, width), dtype=np.float32)
        * (hyperparams.outlier_probability * 1.0 / hyperparams.outlier_volume)
    )
    for gaussian_index in tqdm(range(len(gaussian_xyzs))):
        gaussian_center = gaussian_xyzs[gaussian_index]
        pixel_center = (
            np.array(
                [
                    fy * gaussian_center[1] / gaussian_center[2] + cy,
                    fx * gaussian_center[0] / gaussian_center[2] + cx,
                ]
            )
            .round()
            .astype(np.int32)
        )
        for i in range(
            pixel_center[0] - hyperparams.window_half_width,
            pixel_center[0] + hyperparams.window_half_width + 1,
        ):
            for j in range(
                pixel_center[1] - hyperparams.window_half_width,
                pixel_center[1] + hyperparams.window_half_width + 1,
            ):
                if i >= 0 and i < height and j >= 0 and j < width:
                    observed_rgb_pixel = rgb[i, j]
                    observed_depth_pixel = depth[i, j]
                    observed_xyz_pixel = np.array(
                        [
                            (j - cx) * observed_depth_pixel / fx,
                            (i - cy) * observed_depth_pixel / fy,
                            observed_depth_pixel,
                        ],
                        dtype=np.float32,
                    )

                    difference = np.concatenate(
                        [observed_xyz_pixel, observed_rgb_pixel]
                    ) - np.concatenate(
                        [
                            gaussian_xyzs[gaussian_index],
                            gaussian_rgbs[gaussian_index],
                        ]
                    )
                    component_scores = scipy.stats.multivariate_normal.logpdf(
                        difference, cov=covariances_np[gaussian_index]
                    )
                    expected_score = (
                        component_scores
                        - np.log(number_of_gaussians)
                        + np.log(1.0 - hyperparams.outlier_probability)
                    )
                    log_score_image_reconstruction[i, j] = np.logaddexp(
                        log_score_image_reconstruction[i, j], expected_score
                    )
    match = np.allclose(log_score_image_np, log_score_image_reconstruction, atol=5e-3)
    print(match)
    assert match, np.abs(log_score_image_np - log_score_image_reconstruction).max()
    assert np.all(~np.isnan(log_score_image_np))

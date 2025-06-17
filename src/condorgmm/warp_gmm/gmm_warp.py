import warp as wp
import numpy as np
import jax.numpy as jnp
import rerun as rr
from condorgmm import Pose
import condorgmm


@wp.struct
class GMM_Warp:
    spatial_means: wp.array(dtype=wp.vec3)  # (N, 3)
    rgb_means: wp.array(dtype=wp.vec3)  # (N, 3)
    log_spatial_scales: wp.array(dtype=wp.vec3)  # (N, 3)
    log_rgb_scales: wp.array(dtype=wp.vec3)  # (N, 3)
    quaternions_imaginary: wp.array(dtype=wp.vec3)  # (N, 3)
    quaternions_real: wp.array(dtype=wp.float32)  # (N, 1)
    log_mixture_weights: wp.array(dtype=wp.float32)  # (N, )

    assignments: wp.array(dtype=wp.int32)
    camera_posquat: wp.array(dtype=wp.float32)
    object_posquats: wp.array(ndim=2, dtype=wp.float32)

    def is_valid(self):
        return ~(
            np.isnan(self.spatial_means.numpy()).any()
            or np.isnan(self.rgb_means.numpy()).any()
            or np.isnan(self.log_spatial_scales.numpy()).any()
            or np.isnan(self.log_rgb_scales.numpy()).any()
            or np.isnan(self.quaternions_imaginary.numpy()).any()
            or np.isnan(self.quaternions_real.numpy()).any()
            or np.isnan(self.log_mixture_weights.numpy()).any()
            or np.isnan(self.assignments.numpy()).any()
            or np.isnan(self.camera_posquat.numpy()).any()
            or np.isinf(self.spatial_means.numpy()).any()
            or np.isinf(self.rgb_means.numpy()).any()
            or np.isinf(self.log_spatial_scales.numpy()).any()
            or np.isinf(self.log_rgb_scales.numpy()).any()
            or np.isinf(self.quaternions_imaginary.numpy()).any()
            or np.isinf(self.quaternions_real.numpy()).any()
            or np.isinf(self.log_mixture_weights.numpy()).any()
            or np.isinf(self.assignments.numpy()).any()
            or np.isinf(self.camera_posquat.numpy()).any()
        )


def concatenate_gmms(gmms):
    spatial_means = np.concatenate([gmm.spatial_means.numpy() for gmm in gmms], axis=0)
    rgb_means = np.concatenate([gmm.rgb_means.numpy() for gmm in gmms], axis=0)
    log_spatial_scales = np.concatenate(
        [gmm.log_spatial_scales.numpy() for gmm in gmms], axis=0
    )
    log_rgb_scales = np.concatenate(
        [gmm.log_rgb_scales.numpy() for gmm in gmms], axis=0
    )
    quaternions_imaginary = np.concatenate(
        [gmm.quaternions_imaginary.numpy() for gmm in gmms], axis=0
    )
    quaternions_real = np.concatenate(
        [gmm.quaternions_real.numpy() for gmm in gmms], axis=0
    )
    log_mixture_weights = np.concatenate(
        [gmm.log_mixture_weights.numpy() for gmm in gmms], axis=0
    )

    assignments = np.concatenate(
        [np.full((gmm.spatial_means.shape[0],), i) for i, gmm in enumerate(gmms)],
        axis=0,
    )
    camera_posquat = gmms[0].camera_posquat.numpy()
    object_posquats = np.concatenate(
        [gmm.object_posquats.numpy() for gmm in gmms], axis=0
    )

    return gmm_warp_constructor(
        wp.array(spatial_means, dtype=wp.vec3),
        wp.array(rgb_means, dtype=wp.vec3),
        wp.array(log_spatial_scales, dtype=wp.vec3),
        wp.array(log_rgb_scales, dtype=wp.vec3),
        wp.array(quaternions_imaginary, dtype=wp.vec3),
        wp.array(quaternions_real, dtype=wp.float32),
        wp.array(log_mixture_weights, dtype=wp.float32),
        wp.array(assignments, dtype=wp.int32),
        wp.array(camera_posquat, dtype=wp.float32),
        wp.array(object_posquats, dtype=wp.float32),
    )




def gmm_warp_constructor(
    spatial_means,
    rgb_means,
    log_spatial_scales,
    log_rgb_scales,
    quaternions_imaginary,
    quaternions_real,
    log_mixture_weights=None,
    assignments=None,
    camera_posquat=None,
    object_posquats=None,
):
    gmm = GMM_Warp()
    gmm.spatial_means = spatial_means
    gmm.rgb_means = rgb_means
    gmm.log_spatial_scales = log_spatial_scales
    gmm.log_rgb_scales = log_rgb_scales
    gmm.quaternions_imaginary = quaternions_imaginary
    gmm.quaternions_real = quaternions_real

    if log_mixture_weights is None:
        log_mixture_weights = wp.zeros(spatial_means.shape[0], dtype=wp.float32)
    if assignments is None:
        assignments = wp.zeros(spatial_means.shape[0], dtype=wp.int32)
    if camera_posquat is None:
        camera_posquat = wp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=wp.float32)
    if object_posquats is None:
        object_posquats = wp.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=wp.float32
        )

    gmm.log_mixture_weights = log_mixture_weights
    gmm.assignments = assignments
    gmm.camera_posquat = camera_posquat
    gmm.object_posquats = object_posquats
    return gmm


def gmm_warp_from_gmm_jax(gmm_jax, reinitialize=False):
    if not reinitialize:
        spatial_means = wp.from_jax(gmm_jax.spatial_means, dtype=wp.vec3)
        rgb_means = wp.from_jax(gmm_jax.rgb_means.astype(jnp.float32), dtype=wp.vec3)
        log_spatial_scales = wp.from_jax(jnp.log(gmm_jax.spatial_scales), dtype=wp.vec3)
        log_rgb_scales = wp.from_jax(jnp.log(gmm_jax.rgb_scales), dtype=wp.vec3)
        quaternions_imaginary = wp.from_jax(gmm_jax.quats[:, :3], dtype=wp.vec3)
        quaternions_real = wp.from_jax(gmm_jax.quats[:, 3], dtype=wp.float32)
        log_mixture_weights = wp.from_jax(jnp.log(gmm_jax.probs), dtype=wp.float32)
    else:
        spatial_means = wp.array(np.array(gmm_jax.spatial_means), dtype=wp.vec3)
        rgb_means = wp.array(np.array(gmm_jax.rgb_means), dtype=wp.vec3)
        log_spatial_scales = wp.array(
            np.log(np.array(gmm_jax.spatial_scales)), dtype=wp.vec3
        )
        log_rgb_scales = wp.array(np.log(np.array(gmm_jax.rgb_scales)), dtype=wp.vec3)
        quaternions_imaginary = wp.array(np.array(gmm_jax.quats[:, :3]), dtype=wp.vec3)
        quaternions_real = wp.array(np.array(gmm_jax.quats[:, 3]), dtype=wp.float32)
        log_mixture_weights = wp.array(
            np.log(np.array(gmm_jax.probs)), dtype=wp.float32
        )
    gmm = gmm_warp_constructor(
        spatial_means=spatial_means,
        rgb_means=rgb_means,
        log_spatial_scales=log_spatial_scales,
        log_rgb_scales=log_rgb_scales,
        quaternions_imaginary=quaternions_imaginary,
        quaternions_real=quaternions_real,
        log_mixture_weights=log_mixture_weights,
    )
    return gmm


def gmm_warp_from_numpy(
    spatial_means,
    rgb_means,
    log_spatial_scales=None,
    log_rgb_scales=None,
    quaternions_imaginary=None,
    quaternions_real=None,
    log_mixture_weights=None,
    camera_posquat=None,
    object_posquats=None,
):
    if quaternions_imaginary is None:
        quaternions_imaginary = np.zeros((spatial_means.shape[0], 3), dtype=np.float32)
    if quaternions_real is None:
        quaternions_real = np.ones(spatial_means.shape[0], dtype=np.float32)
    if log_mixture_weights is None:
        log_mixture_weights = np.zeros(
            (spatial_means.shape[0],), dtype=np.float32
        ) - np.log(spatial_means.shape[0])

    if log_spatial_scales is None:
        log_spatial_scales = np.log(
            0.01 * np.ones((spatial_means.shape[0], 3), dtype=np.float32)
        )
    if log_rgb_scales is None:
        log_rgb_scales = np.log(
            5.0 * np.ones((spatial_means.shape[0], 3), dtype=np.float32)
        )
    if camera_posquat is None:
        camera_posquat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    if object_posquats is None:
        object_posquats = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        )

    assert spatial_means.dtype == np.float32
    assert rgb_means.dtype == np.float32
    assert log_spatial_scales.dtype == np.float32
    assert log_rgb_scales.dtype == np.float32
    assert quaternions_imaginary.dtype == np.float32
    assert quaternions_real.dtype == np.float32
    assert log_mixture_weights.dtype == np.float32

    spatial_means = wp.array(spatial_means, dtype=wp.vec3)
    rgb_means = wp.array(rgb_means, dtype=wp.vec3)
    log_spatial_scales = wp.array(log_spatial_scales, dtype=wp.vec3)
    log_rgb_scales = wp.array(log_rgb_scales, dtype=wp.vec3)
    quaternions_imaginary = wp.array(quaternions_imaginary, dtype=wp.vec3)
    quaternions_real = wp.array(quaternions_real, dtype=wp.float32)
    log_mixture_weights = wp.array(log_mixture_weights, dtype=wp.float32)
    camera_posquat = wp.array(camera_posquat, dtype=wp.float32)
    object_posquats = wp.array(object_posquats, dtype=wp.float32)

    gmm = gmm_warp_constructor(
        spatial_means=spatial_means,
        rgb_means=rgb_means,
        log_spatial_scales=log_spatial_scales,
        log_rgb_scales=log_rgb_scales,
        quaternions_imaginary=quaternions_imaginary,
        quaternions_real=quaternions_real,
        log_mixture_weights=log_mixture_weights,
        camera_posquat=camera_posquat,
        object_posquats=object_posquats,
    )
    return gmm


def rr_log_gmm_warp(gmm, channel="gmm", fill_mode=None, size_scalar=8.0, colors=None):
    # log

    camera_pose = gmm.camera_posquat.numpy()
    camera_pose = Pose(camera_pose)

    condorgmm.rr_log_pose(camera_pose, channel + "/camera_pose")

    assignments = gmm.assignments.numpy()

    unique_assignments = np.unique(assignments)
    for assignment in unique_assignments:
        mask = assignments == assignment
        object_pose = gmm.object_posquats.numpy()[assignment]
        object_pose = Pose(object_pose)
        condorgmm.rr_log_pose(object_pose, channel + f"/object_pose_{assignment}", radii=0.001)

        assignment_channel = channel + f"/_{assignment}"
        rr.log(
            assignment_channel,
            rr.Ellipsoids3D(
                centers=gmm.spatial_means.numpy()[mask],
                half_sizes=np.exp(gmm.log_spatial_scales.numpy())[mask] * size_scalar,
                quaternions=np.hstack(
                    [
                        gmm.quaternions_imaginary.numpy()[mask],
                        gmm.quaternions_real.numpy()[mask].reshape(-1, 1),
                    ]
                ),
                colors=colors[mask] if colors is not None else gmm.rgb_means.numpy()[mask] / 255.0,
                fill_mode=fill_mode,
            ),
        )
        rr.log(
            assignment_channel,
            rr.Transform3D(
                translation=object_pose.posquat[:3],
                quaternion=object_pose.posquat[3:],
            ),
        )

def mask_gmm(gmm, mask):
    return gmm_warp_constructor(
        wp.array(gmm.spatial_means.numpy()[mask], dtype=wp.vec3),
        wp.array(gmm.rgb_means.numpy()[mask], dtype=wp.vec3),
        wp.array(gmm.log_spatial_scales.numpy()[mask], dtype=wp.vec3),
        wp.array(gmm.log_rgb_scales.numpy()[mask], dtype=wp.vec3),
        wp.array(gmm.quaternions_imaginary.numpy()[mask], dtype=wp.vec3),
        wp.array(gmm.quaternions_real.numpy()[mask], dtype=wp.float32),
        wp.array(gmm.log_mixture_weights.numpy()[mask], dtype=wp.float32),
        wp.array(gmm.assignments.numpy()[mask], dtype=wp.int32),
        wp.array(gmm.camera_posquat.numpy(), dtype=wp.float32),
        wp.array(gmm.object_posquats.numpy(), dtype=wp.float32),
    )
    
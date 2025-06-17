from __future__ import annotations
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as Rot
from .pose import Pose

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Intrinsics


def xyz_to_cameraxyd(xyz: jnp.ndarray, intr: Intrinsics):
    x = intr.fx * xyz[..., 0] / xyz[..., 2]
    y = intr.fy * xyz[..., 1] / xyz[..., 2]
    d = xyz[..., 2]
    return jnp.stack([x, y, d], axis=-1)


def xyz_from_cameraxyd(camera_xyd: jnp.ndarray, intr: Intrinsics):
    u, v, d = camera_xyd[..., 0], camera_xyd[..., 1], camera_xyd[..., 2]
    x = u * d / intr.fx
    y = v * d / intr.fy
    z = d
    return jnp.stack([x, y, z], axis=-1)


def cov_to_isovars_and_quaternion(cov):
    eigvals, eigvecs = jnp.linalg.eigh(cov)

    # Ensure positive eigenvalues (numerical stability)
    vars = jnp.maximum(eigvals, 0)

    # Ensure deterministic eigenvector orientation
    for i in range(3):
        eigvecs = eigvecs.at[:, i].set(
            jnp.where(eigvecs[0, i] < 0, -eigvecs[:, i], eigvecs[:, i])
        )

    # Ensure proper rotation matrix (determinant +1)
    eigvecs = eigvecs.at[:, 0].set(
        jnp.where(jnp.linalg.det(eigvecs) < 0, -eigvecs[:, 0], eigvecs[:, 0])
    )

    # Convert rotation matrix to quaternion
    quat = Rot.from_matrix(eigvecs).as_quat()

    return vars, quat


def isovars_and_quaternion_to_cov(vars, quat):
    rot = Rot.from_quat(quat).as_matrix()
    cov = rot @ jnp.diag(vars) @ rot.T
    return cov


def cov_to_isostds_and_quaternion(cov):
    vars, quat = cov_to_isovars_and_quaternion(cov)
    return jnp.sqrt(vars), quat


def isostds_and_quaternion_to_cov(stds, quat):
    return isovars_and_quaternion_to_cov(stds**2, quat)


def find_aligning_pose(
    xyz_0: jnp.ndarray,
    xyz_1: jnp.ndarray,
    mask: jnp.ndarray,
) -> Pose:
    X = jnp.where(mask[:, None], xyz_0, jnp.zeros_like(xyz_0))
    Y = jnp.where(mask[:, None], xyz_1, jnp.zeros_like(xyz_1))

    # Compute centroids.
    centroid_X = jnp.sum(X, axis=0) / jnp.sum(mask)
    centroid_Y = jnp.sum(Y, axis=0) / jnp.sum(mask)

    # Center the points.
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    # Compute the cross-covariance matrix.
    H = X_centered.T @ Y_centered

    # Perform SVD on the covariance matrix.
    U, _, Vt = jnp.linalg.svd(H)

    # Compute rotation matrix.
    R = Vt.T @ U.T

    # Reflection correction: ensure a proper rotation (determinant should be +1).
    # if float(jnp.linalg.det(R)) < 0:
    #     Vt = Vt.at[-1].multiply(-1)
    #     R = Vt.T @ U.T
    Vt = jnp.where(jnp.linalg.det(R) < 0, Vt.at[-1].multiply(-1), Vt)
    R = jnp.where(jnp.linalg.det(R) < 0, Vt.T @ U.T, R)

    # Compute the translation vector.
    t = centroid_Y - R @ centroid_X

    return Pose.from_pos_matrix(pos=t, matrix=R)

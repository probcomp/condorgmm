import jax.random as r
from jax.random import key as prngkey
import jax.numpy as jnp
from condorgmm.condor.types import Intrinsics, Pose
from condorgmm.condor.geometry import (
    xyz_to_cameraxyd,
    xyz_from_cameraxyd,
    find_aligning_pose,
)
import pytest


def _generate_intrinsics(key):
    k1, k2, k3, k4, k5 = r.split(key, 5)
    fx = r.uniform(k1, shape=(), minval=1.0, maxval=300.0)
    fy = fx
    image_width = int(r.randint(k3, (), 100, 1000))
    image_height = int(r.randint(k4, (), 100, 1000))
    cx = r.uniform(k2, shape=(), minval=0.0, maxval=image_width)
    cy = r.uniform(k5, shape=(), minval=0.0, maxval=image_height)
    return Intrinsics(
        fx, fy, cx, cy, jnp.array(1e-5), jnp.array(1e5), image_height, image_width
    )


def _generate_camera_xyd(key, intrinsics):
    k1, k2, k3 = r.split(key, 3)
    x = r.uniform(k1, shape=(), minval=0.0, maxval=intrinsics.image_width)
    y = r.uniform(k2, shape=(), minval=0.0, maxval=intrinsics.image_height)
    d = r.uniform(k3, shape=(), minval=1e-5, maxval=1e5)
    return jnp.array([x, y, d])


def check_xyz_round_trip(camera_xyd, intr):
    """
    Check that the round trip from camera_xyd to camera_xyz and back to camera_xyd is the identity.
    """
    xyz = xyz_from_cameraxyd(camera_xyd, intr)
    xyd_reconstructed = xyz_to_cameraxyd(xyz, intr)
    assert jnp.allclose(camera_xyd, xyd_reconstructed)


@pytest.mark.parametrize("key", [prngkey(i) for i in range(10)])
def test_xyz_round_trips(key):
    k1, k2 = r.split(key, 2)
    intr = _generate_intrinsics(k1)
    camera_xyd = _generate_camera_xyd(k2, intr)
    check_xyz_round_trip(camera_xyd, intr)


### Test find_aligning_pose ###


def _generate_zero_error_pose_alignment_problem(key):
    k1, k2, k3, k4, k5 = r.split(key, 5)
    xyz_0 = r.uniform(k1, shape=(100, 3), minval=-10.0, maxval=10.0)
    pose_position = r.uniform(k2, shape=(3,), minval=-10.0, maxval=10.0)
    quat = r.normal(k3, shape=(4,))
    quat = quat / jnp.linalg.norm(quat)
    pose = Pose(position=pose_position, quaternion=quat)
    xyz_1 = pose.apply(xyz_0)

    # randomly mask out 20%
    mask = r.bernoulli(k4, p=1.0, shape=(100,))
    xyz_1 = jnp.where(
        mask[:, None], xyz_1, r.uniform(k5, shape=(100, 3), minval=-10.0, maxval=10.0)
    )

    return xyz_0, xyz_1, mask


def _generate_noisy_pose_alignment_problem(key):
    k1, k2, k3, k4 = r.split(key, 4)
    xyz_0, xyz_1, mask = _generate_zero_error_pose_alignment_problem(k1)
    noise_level = r.exponential(k2, shape=()) * 0.01
    tolerable_error = noise_level * 10
    xyz_0 = xyz_0 + r.normal(k4, shape=(100, 3)) * noise_level / 2
    xyz_1 = xyz_1 + r.normal(k3, shape=(100, 3)) * noise_level / 2
    return xyz_0, xyz_1, mask, tolerable_error


@pytest.mark.parametrize("key", [prngkey(i) for i in range(10)])
def test_find_aligning_pose(key):
    xyz_0, xyz_1, mask = _generate_zero_error_pose_alignment_problem(key)
    pose = find_aligning_pose(xyz_0, xyz_1, mask)
    assert jnp.allclose(pose.apply(xyz_0[mask]), xyz_1[mask], atol=1e-5)

    xyz_0, xyz_1, mask, tolerable_error = _generate_noisy_pose_alignment_problem(key)
    pose = find_aligning_pose(xyz_0, xyz_1, mask)
    assert jnp.allclose(pose.apply(xyz_0[mask]), xyz_1[mask], atol=tolerable_error)

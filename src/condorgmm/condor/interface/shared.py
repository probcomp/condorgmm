import jax
import jax.numpy as jnp
import condorgmm
from ..config import DEFAULT_HYPERPARAMS
from ..types import Intrinsics, Observation, Gaussian, Hyperparams
from ..pose import Pose as CondorPose
from ..geometry import cov_to_isostds_and_quaternion
from functools import partial


@partial(jax.jit, static_argnames=("height", "width"))
def _get_dp_mask(height, width):
    return jnp.ones(height * width, dtype=bool)


@partial(jax.jit, static_argnames=("height", "width"))
def _to_observations_inner(
    rgb,
    depth,
    hypers,
    key,
    cx,
    cy,
    height,
    width,
):
    ys, xs = jnp.mgrid[:height, :width]
    ys, xs = ys.flatten(), xs.flatten()
    xys = jnp.stack([xs, ys], axis=-1)
    xys = jnp.array(xys + 0.5 - jnp.array([cx, cy]), dtype=jnp.float32)

    rgb = jnp.array(rgb.reshape(-1, 3), dtype=jnp.float32)
    rgb = (
        rgb
        + jax.random.normal(key, rgb.shape, dtype=jnp.float32)
        * hypers.rgb_noisefloor_std
    )
    return Observation(
        rgb=rgb,
        camera_xy=xys,
        depth=jnp.array(depth.reshape(-1), dtype=jnp.float32),
    )


def _to_observations(
    frame: condorgmm.Frame, hypers: Hyperparams = DEFAULT_HYPERPARAMS, key=jax.random.key(0)
) -> Observation:
    _, _, cx, cy = frame.intrinsics
    return _to_observations_inner(
        frame.rgb, frame.depth, hypers, key, cx, cy, frame.height, frame.width
    )


@jax.jit
def _to_condor_pose_inner(pos, xyzw):
    return CondorPose(jnp.array(pos, dtype=float), jnp.array(xyzw, dtype=float))


def _to_condor_pose(pose: condorgmm.Pose) -> CondorPose:
    return _to_condor_pose_inner(pose.pos, pose.xyzw)


def _f(x):
    return jnp.array(x, dtype=jnp.float32)


def _frame_to_intrinsics(
    frame: condorgmm.Frame, near=1e-4, far=1e4, ensure_fx_eq_fy: bool = True
) -> Intrinsics:
    fx, fy, cx, cy = frame.intrinsics
    if ensure_fx_eq_fy:
        assert jnp.allclose(
            fx, fy, rtol=1e-2
        ), "Depth inference code is currently not set up to work with differing fx and fy."
        fy = fx
    return Intrinsics(
        _f(fx), _f(fy), _f(cx), _f(cy), _f(near), _f(far), frame.height, frame.width
    )


@jax.jit
def _to_gmm_inner(gaussian: Gaussian):
    isostds, quats = jax.vmap(cov_to_isostds_and_quaternion)(gaussian.xyz_cov)
    return (
        gaussian.xyz,
        quats,
        jnp.clip(gaussian.rgb, 0, 255).astype(jnp.int32),
        isostds,
        jnp.sqrt(gaussian.rgb_vars),
        (gaussian.mixture_weight / jnp.sum(gaussian.mixture_weight)),
    )


def _to_gmm(gaussian: Gaussian) -> condorgmm.GMM:
    # spatial_means, quats, rgb_means, spatial_scales, rgb_scales, probs = _to_gmm_inner(
    #     gaussian
    # )
    return condorgmm.GMM(*_to_gmm_inner(gaussian))

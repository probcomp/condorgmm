import jax
import jax.numpy as jnp
import numpy as np
import rerun as rr
from genjax import Mask
import condorgmm.condor.types as t
import warnings
from .geometry import cov_to_isostds_and_quaternion
from .model.model import generate_datapoint, n_background_gaussians
from .tiling import MonolithicTiling, GridTiling
from genjax.typing import BoolArray
from typing import cast


def _ellipsoids(
    gaussians: t.Gaussian,
    mask: jnp.ndarray | None = None,
    std_scalar: float = 1,
    fill_mode: rr.components.FillMode = rr.components.FillMode.MajorWireframe,
    pose_transform=t.Pose.identity(),
    do_color=True,
    class_ids: jnp.ndarray | None = None,
):
    if mask is None:
        mask = jnp.ones(len(gaussians), dtype=bool)
    iso_stds, quats = jax.vmap(cov_to_isostds_and_quaternion)(gaussians.xyz_cov)
    half_sizes = iso_stds * std_scalar

    og_poses = t.Pose(gaussians.xyz, quats)
    transformed = pose_transform @ og_poses
    xyz = transformed.pos
    quats = transformed.quaternion
    return rr.Ellipsoids3D(
        centers=np.array(
            jnp.where(mask[:, None], xyz, jnp.nan * jnp.ones_like(gaussians.xyz))
        ),
        quaternions=np.array(quats),
        half_sizes=np.array(half_sizes),
        colors=(np.array(jnp.clip(gaussians.rgb / 255, 0, 1)) if do_color else None),
        class_ids=class_ids,
        fill_mode=fill_mode,
    )


def _log_gaussians(
    gaussians: t.Gaussian,
    gaussian_has_assoc_mask,
    pose_transform,
    channel="gaussians",
    fill_mode=rr.components.FillMode.MajorWireframe,
):
    rr.log(
        f"{channel}/all",
        _ellipsoids(gaussians, pose_transform=pose_transform, fill_mode=fill_mode),
    )
    rr.log(
        f"{channel}/with_assoc",
        _ellipsoids(
            gaussians,
            gaussian_has_assoc_mask,
            pose_transform=pose_transform,
            fill_mode=fill_mode,
        ),
    )
    rr.log(
        f"{channel}/persisting_and_with_assoc",
        _ellipsoids(
            gaussians,
            jnp.logical_and(
                gaussian_has_assoc_mask,
                gaussians.origin != -1,
            ),
            pose_transform=pose_transform,
            fill_mode=fill_mode,
        ),
    )


def log_gaussians(
    st: t.CondorGMMState,
    hypers: t.Hyperparams,
    log_in_world_frame=False,
    channel="gaussians",
    ellipse_mode: rr.components.FillMode = rr.components.FillMode.MajorWireframe,
    ellipse_scalar: float = 1,
    filter_long_gaussians=False,
):
    if log_in_world_frame:
        pose_transform = st.scene.transform_World_Camera
    else:
        pose_transform = t.Pose.identity()

    gaussians = st.gaussians.replace(xyz_cov=st.gaussians.xyz_cov * ellipse_scalar**2)

    isostds, _ = jax.vmap(cov_to_isostds_and_quaternion)(gaussians.xyz_cov)
    max_isostd = jnp.max(isostds, axis=-1)
    is_long = max_isostd > 0.05

    mask = st.gaussian_has_assoc_mask[: n_background_gaussians(hypers)]

    if filter_long_gaussians:
        mask = jnp.logical_and(
            mask,
            jnp.logical_not(is_long)[: n_background_gaussians(hypers)],
        )

    _log_gaussians(
        gaussians[: n_background_gaussians(hypers)],
        mask,
        pose_transform,
        channel=f"{channel}/background",
        fill_mode=ellipse_mode,
    )
    if isinstance(st.scene, t.SingleKnownObjectSceneState):
        _log_gaussians(
            st.gaussians[n_background_gaussians(hypers) :],
            st.gaussian_has_assoc_mask[n_background_gaussians(hypers) :],
            pose_transform,
            channel=f"{channel}/object",
        )


def _log_datapoints(
    datapoints: Mask[t.Datapoint], channel, pose_transform=t.Pose.identity()
):
    flag = cast(BoolArray, datapoints.flag)
    xyz = jnp.where(flag[:, None], datapoints.value.xyz, jnp.nan)
    xyz = pose_transform.apply(xyz)
    rgb = jnp.clip(datapoints.value.rgb / 255, 0, 1)
    rr.log(channel, rr.Points3D(np.array(xyz), colors=np.array(rgb)))


def log_datapoints(
    st: t.CondorGMMState,
    hypers: t.Hyperparams,
    channel="datapoints",
    log_in_world_frame=False,
):
    if log_in_world_frame:
        pose_transform = st.scene.transform_World_Camera
    else:
        pose_transform = t.Pose.identity()
    has_depth_return = st.datapoints.value.obs.depth > 0
    _log_datapoints(
        st.datapoints[has_depth_return],
        f"{channel}/with_observed_depth",
        pose_transform,
    )
    _log_datapoints(
        st.datapoints[~has_depth_return],
        f"{channel}/with_inferred_depth",
        pose_transform,
    )


@jax.jit
def _simulate_reconstruction(key, i, st, tiling, hypers):
    return generate_datapoint.simulate(
        key, (i, st.matter.replace(tiling=tiling), hypers)
    ).get_retval()


printed_note_about_warning = False


def log_reconstruction(
    st: t.CondorGMMState,
    hypers: t.Hyperparams,
    sqrt_density_increase=2,
    log_in_world_frame=False,
):
    if log_in_world_frame:
        pose_transform = st.scene.transform_World_Camera
    else:
        pose_transform = t.Pose.identity()

    # Change the tiling to increase the density at which points will be generated.
    if isinstance(st.matter.tiling, GridTiling):
        tiling: GridTiling = st.matter.tiling
        new_config = tiling.config.replace(
            intrinsics=tiling.config.intrinsics.upscale(sqrt_density_increase),
            tile_size_x=tiling.config.tile_size_x * sqrt_density_increase,
            tile_size_y=tiling.config.tile_size_y * sqrt_density_increase,
        )
        global printed_note_about_warning
        if not printed_note_about_warning:
            print(
                "Note: If a warning from Intrinsics.upscale was emitted, it is for a computation in the rerun logging, so will not quantitative accuracy."
            )
            printed_note_about_warning = True
        new_tiling = tiling.replace(config=new_config)
        n_datapoints = len(hypers.datapoint_mask) * sqrt_density_increase**2
    elif isinstance(st.matter.tiling, MonolithicTiling):
        new_tiling = st.matter.tiling.replace(
            n_datapoints=st.matter.tiling.n_datapoints * sqrt_density_increase**2
        )
        n_datapoints = len(hypers.datapoint_mask) * sqrt_density_increase**2
    else:
        warnings.warn(
            "Unrecognized tiling type; log_reconstruction will not change the density at which points are generated."
        )
        new_tiling = st.matter.tiling
        n_datapoints = len(hypers.datapoint_mask)

    datapoints = jax.vmap(
        lambda k, i: _simulate_reconstruction(k, i, st, new_tiling, hypers)
    )(jax.random.split(jax.random.key(0), n_datapoints), jnp.arange(n_datapoints))
    flag = st.gaussian_has_assoc_mask[datapoints.gaussian_idx]
    _log_datapoints(
        Mask(datapoints, flag),
        "reconstruction",
        pose_transform=pose_transform,
    )


def log_state(
    st: t.CondorGMMState,
    hypers: t.Hyperparams,
    log_in_world_frame=False,
    ellipse_mode: rr.components.FillMode = rr.components.FillMode.Solid,
    ellipse_scalar: float = 2,
):
    log_gaussians(
        st,
        hypers,
        log_in_world_frame=log_in_world_frame,
        ellipse_mode=ellipse_mode,
        ellipse_scalar=ellipse_scalar,
    )
    log_datapoints(st, hypers, log_in_world_frame=log_in_world_frame)
    log_reconstruction(st, hypers, log_in_world_frame=log_in_world_frame)

import jax
from jax.random import split
import jax.numpy as jnp
import condorgmm
from genjax import Pytree, Mask
from trimesh import Trimesh
import condorgmm.condor.inference.frame0 as frame0
import condorgmm.condor.inference.step as step
from ..pose import Pose
from ..inference.instrumentation import InferenceLoggingConfig, Metadata
from ..types import (
    CondorGMMState,
    Hyperparams,
    Gaussian,
    BackgroundOnlySceneState,
    SingleKnownObjectSceneState,
    Datapoint,
)
from ..utils import MyPytree
from ..config import DEFAULT_HYPERPARAMS
from .shared import (
    _to_observations,
    _to_condor_pose,
    _to_gmm,
    _frame_to_intrinsics,
    _get_dp_mask,
)
from typing import Tuple, Any, cast
import condorgmm.condor.model.model as model
from functools import partial


@Pytree.dataclass
class CondorObjectTrackingConfig(MyPytree):
    n_pts_for_object_fitting: int = Pytree.static()
    n_sweeps_per_phase_for_object_fitting: Tuple[int, ...] = Pytree.static()
    do_reinitialize_per_phase_for_object_fitting: Tuple[bool, ...] = Pytree.static()

    n_sweeps_per_phase: Tuple[int, ...] = Pytree.static()
    do_reinitialize_per_phase: Tuple[bool, ...] = Pytree.static()
    phase_to_add_depth_non_returns: int = Pytree.static()

    n_sweeps_after_adding_object: int = Pytree.static()

    step_run_second_pass: bool = Pytree.static()
    step_n_sweeps_phase_1: int = Pytree.static()
    step_n_sweeps_phase_2: int = Pytree.static()

    n_gaussians_for_object: int = Pytree.static()
    n_gaussians_for_background: int = Pytree.static()
    tile_size_x: int | None = Pytree.static(default=None)
    tile_size_y: int | None = Pytree.static(default=None)
    max_n_gaussians_per_tile: int | None = Pytree.static(default=None)
    repopulate_depth_nonreturns: bool = Pytree.static(default=True)
    base_hypers: Hyperparams = Pytree.field(default=DEFAULT_HYPERPARAMS)


default_cfg = CondorObjectTrackingConfig(
    n_pts_for_object_fitting=10000,
    n_sweeps_per_phase_for_object_fitting=(10, 10, 10),
    do_reinitialize_per_phase_for_object_fitting=(False, True, True),
    n_sweeps_per_phase=(20, 20, 20, 20),
    do_reinitialize_per_phase=(False, True, True, True),
    phase_to_add_depth_non_returns=2,
    n_sweeps_after_adding_object=20,
    step_run_second_pass=True,
    step_n_sweeps_phase_1=1,
    step_n_sweeps_phase_2=4,
    n_gaussians_for_object=160,
    n_gaussians_for_background=540,
    tile_size_x=8,
    tile_size_y=8,
    max_n_gaussians_per_tile=32,
    repopulate_depth_nonreturns=False,
    base_hypers=DEFAULT_HYPERPARAMS,
)


@Pytree.dataclass
class CondorObjectTrackerState(MyPytree):
    key: Any
    state: CondorGMMState
    hypers: Hyperparams


def initialize(
    frame: condorgmm.Frame,
    camera_pose_world_frame: condorgmm.Pose,
    object_pose_world_frame: condorgmm.Pose,
    object_mesh: Trimesh,
    object_mask: jnp.ndarray,  # (H, W) boolean array
    cfg: CondorObjectTrackingConfig,
    *,
    key=jax.random.key(0),
    log=False,
) -> tuple[condorgmm.GMM, condorgmm.GMM, CondorObjectTrackerState]:
    k1, k2, k3, k4, k5, k6 = split(key, 6)

    if cfg.repopulate_depth_nonreturns:
        mask = jnp.ones(frame.width * frame.height, dtype=bool)
    else:
        mask = jnp.array(frame.depth > 0, dtype=bool).flatten()  # type: ignore

    hypers_without_object_model = cfg.base_hypers.replace(
        {
            "n_gaussians": cfg.n_gaussians_for_background + cfg.n_gaussians_for_object,
            "tile_size_x": cfg.tile_size_x,
            "tile_size_y": cfg.tile_size_y,
            "intrinsics": _frame_to_intrinsics(
                frame, ensure_fx_eq_fy=cfg.repopulate_depth_nonreturns
            ),
            "datapoint_mask": mask,
            "max_n_gaussians_per_tile": cfg.max_n_gaussians_per_tile,
            "repopulate_depth_nonreturns": cfg.repopulate_depth_nonreturns,
        },
        do_replace_none=False,
    )

    object_model, obj_model_metadata = fit_object_model(
        k1, object_mesh, hypers_without_object_model, cfg, log
    )

    observations = _to_observations(frame, hypers_without_object_model)

    # Initialize a condor state with the background only.
    hypers_masking_object = hypers_without_object_model.replace(
        n_gaussians=hypers_without_object_model.n_gaussians - len(object_model),
        datapoint_mask=jnp.logical_and(
            hypers_without_object_model.datapoint_mask,
            jnp.logical_not(object_mask.flatten()),
        ),
    )
    bkg_only_st, bkg_only_metadata = frame0.run_inference(
        k2,
        observations,
        hypers_masking_object,
        n_sweeps_per_phase=cfg.n_sweeps_per_phase,
        do_reinitialize_per_phase=cfg.do_reinitialize_per_phase,
        phase_to_add_depth_non_returns=cfg.phase_to_add_depth_non_returns,
        c=InferenceLoggingConfig(log),
    )

    # Then add in the object.
    hypers = hypers_without_object_model.replace(
        initial_scene=SingleKnownObjectSceneState(
            transform_World_Camera=_to_condor_pose(camera_pose_world_frame),
            transform_World_Object=_to_condor_pose(object_pose_world_frame),
            object_model=object_model,
        )
    )

    st1 = _add_object_into_bkg_only_state(k3, bkg_only_st, hypers)
    st2, _ = frame0.update_datapoint_associations_and_depths(
        k4,
        st1,
        hypers.replace(always_accept_assoc_depth_move=True),
        c=InferenceLoggingConfig(False),
    )
    st, final_meta = frame0.run_inference(
        k5,
        observations,
        hypers,
        n_sweeps_per_phase=cfg.n_sweeps_per_phase,
        do_reinitialize_per_phase=cfg.do_reinitialize_per_phase,
        phase_to_add_depth_non_returns=cfg.phase_to_add_depth_non_returns,
        c=InferenceLoggingConfig(log),
        st0=st2,
    )
    # frame0.run_n_mcmc_sweeps_jitted(
    #     k5,
    #     st2,
    #     hypers,
    #     n_mcmc_sweeps=cfg.n_sweeps_after_adding_object,
    #     c=InferenceLoggingConfig(log),
    # )

    ret = (
        _to_gmm(_bkg_gaussians(st, hypers)),
        _to_gmm(_obj_gaussians(st, hypers)),
        CondorObjectTrackerState(k6, st, hypers),
    )

    if not log:
        return ret

    return *ret, {  # type: ignore
        "object_model_metadata": obj_model_metadata,
        "bkg_only_metadata": bkg_only_metadata,
        "hypers_masking_object": hypers_masking_object,
        "final_metadata": final_meta,
        "st_after_adding_in_object": st1,
        "st_after_dp_update": st2,
    }


def update(
    frame: condorgmm.Frame,
    camera_pose_world_frame: condorgmm.Pose,
    object_pose_world_frame: condorgmm.Pose,
    prev_state: CondorObjectTrackerState,
    cfg: CondorObjectTrackingConfig = default_cfg,
    *,
    log=False,
    get_gmm=True,
) -> (
    tuple[condorgmm.GMM | None, condorgmm.GMM | None, CondorObjectTrackerState]
    | tuple[condorgmm.GMM | None, condorgmm.GMM | None, CondorObjectTrackerState, Metadata]
):
    k1, k2 = split(prev_state.key)
    inferred_state, metadata = step.run_inference(
        k1,
        prev_state.state,
        _to_observations(frame),
        prev_state.hypers.replace(
            datapoint_mask=_get_dp_mask(frame.height, frame.width),
        ),
        transform_World_Camera=_to_condor_pose(camera_pose_world_frame),
        transform_World_Object=_to_condor_pose(object_pose_world_frame),
        n_sweeps_first_pass=cfg.step_n_sweeps_phase_1,
        n_sweeps_second_pass=cfg.step_n_sweeps_phase_2,
        run_second_pass=cfg.step_run_second_pass,
        c=InferenceLoggingConfig(log),
    )
    cotstate = CondorObjectTrackerState(k2, inferred_state, prev_state.hypers)

    if get_gmm:
        bkg_gmm = _to_gmm(_bkg_gaussians(inferred_state, prev_state.hypers))
        obj_gmm = _to_gmm(_obj_gaussians(inferred_state, prev_state.hypers))
    else:
        bkg_gmm, obj_gmm = None, None

    if log:
        return bkg_gmm, obj_gmm, cotstate, metadata  # type: ignore

    return bkg_gmm, obj_gmm, cotstate


### Internal logic ###
def fit_object_model(
    key,
    object_mesh: Trimesh,
    hypers: Hyperparams,
    cfg: CondorObjectTrackingConfig,
    log=False,
    k_for_initialization=10,
) -> tuple[Gaussian, dict]:  # returns a GMM, as a batched Gaussian
    k1, k2, k3 = split(key, 3)
    xyz, rgb = condorgmm.mesh.sample_surface_points(
        object_mesh, cfg.n_pts_for_object_fitting
    )
    xyz, rgb = jnp.array(xyz, dtype=jnp.float32), jnp.array(rgb, dtype=jnp.float32)
    rgb = (
        rgb
        + jax.random.normal(k1, rgb.shape, dtype=jnp.float32)
        * hypers.rgb_noisefloor_std
    )
    hypers = hypers.replace(
        n_gaussians=cfg.n_gaussians_for_object,
        datapoint_mask=jnp.ones(cfg.n_pts_for_object_fitting, dtype=bool),
        repopulate_depth_nonreturns=False,
        use_monolithic_tiling=True,
        initial_scene=BackgroundOnlySceneState(
            transform_World_Camera=Pose.identity(),
        ),
    )
    gaussian_idxs = _get_sparse_datapoint_assignment_initialization(
        k2, cfg.n_gaussians_for_object, xyz, rgb, k=k_for_initialization
    )
    datapoints = jax.vmap(Datapoint.from_xyz_rgb, in_axes=(0, 0, 0, None))(
        xyz, rgb, gaussian_idxs, hypers
    )
    st, meta = frame0.run_inference(
        k3,
        datapoints.obs,
        hypers,
        n_sweeps_per_phase=cfg.n_sweeps_per_phase_for_object_fitting,
        do_reinitialize_per_phase=cfg.do_reinitialize_per_phase_for_object_fitting,
        phase_to_add_depth_non_returns=-1,  # value doesn't matter since hypers.repopulate_depth_nonreturns=False
        c=InferenceLoggingConfig(log),
        given_datapoint_assignment=gaussian_idxs,
    )
    gaussians = st.gaussians[st.gaussian_has_assoc_mask]
    return gaussians, {} if not log else {"meta": meta, "hypers": hypers}


@jax.jit
def _add_object_into_bkg_only_state(
    key, bkg_only_state: CondorGMMState, hypers: Hyperparams
) -> CondorGMMState:
    scene = cast(SingleKnownObjectSceneState, hypers.initial_scene)
    assert len(scene.object_model) + len(bkg_only_state.gaussians) == hypers.n_gaussians

    object_gaussians = _object_model_in_camera_frame(
        hypers.initial_scene, hypers
    ).replace(
        idx=jnp.arange(model.n_background_gaussians(hypers), hypers.n_gaussians),
        object_idx=jnp.ones(len(scene.object_model), dtype=jnp.int32),
    )
    object_gaussians = object_gaussians.replace(
        rgb_vars=jnp.maximum(
            jnp.array(64.0**2, dtype=float),
            9.0 * object_gaussians.rgb_vars,
        ),
        mixture_weight=object_gaussians.mixture_weight * 5.0,
    )
    new_gaussians = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y]),
        bkg_only_state.gaussians,
        object_gaussians,
    )
    new_tiling = bkg_only_state.matter.tiling.update_tiling(new_gaussians, key=key)
    return bkg_only_state.replace(
        {
            "matter": {"gaussians": new_gaussians, "tiling": new_tiling},
            "datapoints": Mask(bkg_only_state.datapoints.value, hypers.datapoint_mask),
            "scene": hypers.initial_scene,
        }
    )


@jax.jit
def _object_model_in_camera_frame(
    scene: SingleKnownObjectSceneState, hypers: Hyperparams
) -> Gaussian:
    return jax.vmap(
        lambda i: model.get_object_model_gaussian_in_camera_frame(i, scene, hypers)
    )(jnp.arange(model.n_background_gaussians(hypers), hypers.n_gaussians))


@jax.jit
def _bkg_gaussians(st: CondorGMMState, hypers: Hyperparams) -> Gaussian:
    return st.matter.gaussians[: model.n_background_gaussians(hypers)]


@jax.jit
def _obj_gaussians_inner(
    st: CondorGMMState, hypers: Hyperparams
) -> tuple[Gaussian, jnp.ndarray]:
    gaussians_camera_frame = st.matter.gaussians[model.n_background_gaussians(hypers) :]
    transform_Object_Camera = (
        cast(SingleKnownObjectSceneState, st.scene).transform_World_Object.inv()
        @ st.scene.transform_World_Camera
    )
    gaussians_Obj_frame = gaussians_camera_frame.transform_by(transform_Object_Camera)
    mask = st.gaussian_has_assoc_mask[model.n_background_gaussians(hypers) :]
    return gaussians_Obj_frame, mask


def _obj_gaussians(st: CondorGMMState, hypers: Hyperparams) -> Gaussian:
    gaussians, mask = _obj_gaussians_inner(st, hypers)
    return gaussians


@partial(jax.jit, static_argnames=("k", "n_gaussians"))
def _get_sparse_datapoint_assignment_initialization(key, n_gaussians, xyz, rgb, k=10):
    n_datapoints = xyz.shape[0]
    gaussian_to_datapoint = jax.random.choice(
        key, jnp.arange(n_datapoints), shape=(n_gaussians,), replace=False
    )

    def distance(xyz0, rgb0, xyz1, rgb1):
        # 1 mm xyz ~~ 10 units rgb
        distance_meters = jnp.linalg.norm(xyz0 - xyz1)
        distance_mm = distance_meters * 1000
        distance_rgb = jnp.linalg.norm(rgb0 - rgb1)
        return distance_mm + distance_rgb / 10

    # For each chosen datapoint, find the k closest datapoints
    # and assign them to the same Gaussian.
    dp_to_k_closest = jax.vmap(
        lambda dp_idx: jnp.argsort(
            jax.vmap(lambda i: distance(xyz[dp_idx], rgb[dp_idx], xyz[i], rgb[i]))(
                jnp.arange(n_datapoints)
            )
        )[:k]
    )
    gaussian_to_k_datapoints = dp_to_k_closest(gaussian_to_datapoint)
    dp_to_gaussian = (
        (-jnp.ones(n_datapoints, dtype=jnp.int32))
        .at[jnp.concatenate(gaussian_to_k_datapoints)]
        .set(jnp.repeat(jnp.arange(n_gaussians), k))
    )
    return dp_to_gaussian


def _get_initial_scene_state(
    camera_pose_world_frame: Pose,
    object_pose_world_frame: Pose,
    object_model: Gaussian,
    hypers: Hyperparams,
) -> SingleKnownObjectSceneState:
    return SingleKnownObjectSceneState(
        transform_World_Camera=camera_pose_world_frame,
        transform_World_Object=object_pose_world_frame,
        object_model=object_model,
    )

import jax
from jax.random import split
import jax.numpy as jnp
import condorgmm
from genjax import Pytree
from ..types import CondorGMMState, Hyperparams
from ..utils import MyPytree
from ..config import DEFAULT_HYPERPARAMS
from ..inference.instrumentation import InferenceLoggingConfig, Metadata
from ..inference.frame0 import run_inference as run_frame0_inference
from ..inference.step import run_inference as run_step_inference
from .shared import (
    _to_observations,
    _to_condor_pose,
    _frame_to_intrinsics,
    _to_gmm,
    _get_dp_mask,
)
from typing import Tuple, Any


@Pytree.dataclass
class CondorCameraTrackingConfig(MyPytree):
    n_sweeps_per_phase: Tuple[int, ...] = Pytree.static()
    do_reinitialize_per_phase: Tuple[bool, ...] = Pytree.static()
    phase_to_add_depth_non_returns: int = Pytree.static()

    step_run_second_pass: bool = Pytree.static()
    step_n_sweeps_phase_1: int = Pytree.static()
    step_n_sweeps_phase_2: int = Pytree.static()

    n_gaussians: int | None = Pytree.static(default=None)
    tile_size_x: int | None = Pytree.static(default=None)
    tile_size_y: int | None = Pytree.static(default=None)
    max_n_gaussians_per_tile: int | None = Pytree.static(default=None)
    repopulate_depth_nonreturns: bool = Pytree.static(default=True)
    do_pose_update: bool = Pytree.static(default=False)
    base_hypers: Hyperparams = Pytree.field(default=DEFAULT_HYPERPARAMS)


fast_config = CondorCameraTrackingConfig(
    n_sweeps_per_phase=(20, 20, 20, 20),
    do_reinitialize_per_phase=(False, True, True, True),
    phase_to_add_depth_non_returns=2,
    step_run_second_pass=True,
    step_n_sweeps_phase_1=1,
    step_n_sweeps_phase_2=4,
    n_gaussians=320,
    tile_size_x=8,
    tile_size_y=8,
    max_n_gaussians_per_tile=4,
    repopulate_depth_nonreturns=True,
    base_hypers=DEFAULT_HYPERPARAMS,
)
slow_config = CondorCameraTrackingConfig(
    n_sweeps_per_phase=(20, 20, 20, 20),
    do_reinitialize_per_phase=(False, True, True, True),
    phase_to_add_depth_non_returns=2,
    step_run_second_pass=True,
    step_n_sweeps_phase_1=4,
    step_n_sweeps_phase_2=6,
    n_gaussians=320,
    tile_size_x=16,
    tile_size_y=16,
    max_n_gaussians_per_tile=8,
    repopulate_depth_nonreturns=True,
    base_hypers=DEFAULT_HYPERPARAMS,
)


@Pytree.dataclass
class CondorCameraTrackerState(MyPytree):
    key: Any
    state: CondorGMMState
    hypers: Hyperparams


def initialize(
    frame: condorgmm.Frame,
    camera_pose_world_frame: condorgmm.Pose,
    cfg: CondorCameraTrackingConfig = fast_config,
    key: Any = jax.random.key(0),
    log=False,
) -> (
    tuple[condorgmm.GMM, CondorCameraTrackerState]
    | tuple[condorgmm.GMM, CondorCameraTrackerState, Metadata]
):
    k1, k2, k3 = split(key, 3)
    hypers = cfg.base_hypers.replace(
        {
            "n_gaussians": cfg.n_gaussians,
            "tile_size_x": cfg.tile_size_x,
            "tile_size_y": cfg.tile_size_y,
            "intrinsics": _frame_to_intrinsics(
                frame, ensure_fx_eq_fy=cfg.repopulate_depth_nonreturns
            ),
            "datapoint_mask": jnp.ones(frame.height * frame.width, dtype=bool),
            "initial_scene": {
                "transform_World_Camera": _to_condor_pose(camera_pose_world_frame)
            },
            "max_n_gaussians_per_tile": cfg.max_n_gaussians_per_tile,
            "repopulate_depth_nonreturns": cfg.repopulate_depth_nonreturns,
            "do_pose_update": cfg.do_pose_update,
        },
        do_replace_none=False,
    )
    inferred_condor_state, metadata = run_frame0_inference(
        k1,
        _to_observations(frame, hypers, k2),
        hypers,
        n_sweeps_per_phase=cfg.n_sweeps_per_phase,
        do_reinitialize_per_phase=cfg.do_reinitialize_per_phase,
        phase_to_add_depth_non_returns=cfg.phase_to_add_depth_non_returns,
        c=InferenceLoggingConfig(log),
    )
    gmm = _to_gmm(inferred_condor_state.gaussians)
    cctstate = CondorCameraTrackerState(k3, inferred_condor_state, hypers)
    if log:
        return gmm, cctstate, metadata
    return gmm, cctstate


def update(
    frame: condorgmm.Frame,
    camera_pose_world_frame: condorgmm.Pose,
    prev_state: CondorCameraTrackerState,
    cfg: CondorCameraTrackingConfig = fast_config,
    log=False,
    get_gmm=True,
) -> (
    tuple[condorgmm.GMM | None, CondorCameraTrackerState]
    | tuple[condorgmm.GMM, CondorCameraTrackerState, Metadata]
):
    k1, k2 = split(prev_state.key)
    inferred_state, metadata = run_step_inference(
        k1,
        prev_state.state,
        _to_observations(frame),
        prev_state.hypers.replace(
            datapoint_mask=_get_dp_mask(frame.height, frame.width),
        ),
        transform_World_Camera=_to_condor_pose(camera_pose_world_frame),
        n_sweeps_first_pass=cfg.step_n_sweeps_phase_1,
        n_sweeps_second_pass=cfg.step_n_sweeps_phase_2,
        run_second_pass=cfg.step_run_second_pass,
        c=InferenceLoggingConfig(log),
    )
    cctstate = CondorCameraTrackerState(k2, inferred_state, prev_state.hypers)

    if get_gmm:
        # The get_gmm flag exists since it is a little bit slow
        # to call _to_gmm.
        gmm = _to_gmm(inferred_state.gaussians)
    else:
        gmm = None

    if log:
        return gmm, cctstate, metadata  # type: ignore

    return gmm, cctstate

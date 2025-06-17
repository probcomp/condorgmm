import jax
from trimesh import Trimesh
from dataclasses import dataclass
from condorgmm.data import Frame
from condorgmm import GMM
from condorgmm import Pose, rr_set_time
from condorgmm.condor.interface.object_tracking import (
    CondorObjectTrackerState,
    default_cfg,
    initialize as condor_initialize,
    update as condor_update,
)
from condorgmm.condor.rerun import log_state as log_condor_state
from condorgmm.utils.common.rerun import rr_log_pose, rr_log_frame, rr_log_frustum
import condorgmm.warp_gmm as warp_gmm
import warp as wp
import jax.numpy as jnp


@dataclass
class ObjectTrackerHyperparams:
    matter_update_period: int
    n_gaussians_for_object: int
    n_gaussians_for_background: int
    tile_size_x: int = 16
    tile_size_y: int = 16
    condor_repopulate_depth_nonreturns: bool = False
    use_gt_camera_pose: bool = False

    def get_condor_config(self):
        return default_cfg.replace(
            n_gaussians_for_object=self.n_gaussians_for_object,
            n_gaussians_for_background=self.n_gaussians_for_background,
            tile_size_x=self.tile_size_x,
            tile_size_y=self.tile_size_y,
            repopulate_depth_nonreturns=self.condor_repopulate_depth_nonreturns,
        )


DEFAULT_HYPERS = ObjectTrackerHyperparams(
    matter_update_period=10, n_gaussians_for_object=120, n_gaussians_for_background=540
)


@dataclass
class ObjectTrackerState:
    camera_pose: Pose  # uses jax arrays; in world frame
    object_pose: Pose  # uses jax arrays; in world frame
    condor_state: CondorObjectTrackerState
    warp_gmm_state: warp_gmm.State
    object_idx: int
    did_matter_update: bool


def initialize(
    frame: Frame,
    object_mesh: Trimesh,
    object_idx=0,
    hypers=DEFAULT_HYPERS,
    debug=False,
    seed=0,
):
    transform_World_Camera = Pose(jnp.array(frame.camera_pose))
    transform_World_Object = Pose(jnp.array(frame.object_poses[object_idx]))
    (background_gmm_camera_frame, object_gmm_object_frame, condor_state_0) = (
        condor_initialize(
            frame,
            transform_World_Camera,
            transform_World_Object,
            object_mesh,
            frame.masks[object_idx],
            hypers.get_condor_config(),
            key=jax.random.key(seed),
        )
    )
    background_gmm0_world_frame = background_gmm_camera_frame.transform_by(
        transform_World_Camera
    )

    assert isinstance(object_gmm_object_frame, GMM)
    assert isinstance(background_gmm0_world_frame, GMM)

    warp_gmm_object = warp_gmm.gmm_warp_from_gmm_jax(object_gmm_object_frame)
    warp_gmm_object.camera_posquat = wp.from_jax(transform_World_Camera.posquat)
    warp_gmm_object.object_posquats = wp.array(
        transform_World_Object.posquat[None, :], dtype=wp.float32
    )
    warp_gmm_state = warp_gmm.initialize_state(gmm=warp_gmm_object, frame=frame)
    warp_gmm_state.hyperparams.window_half_width = 32

    return (
        transform_World_Camera,
        transform_World_Object,
        ObjectTrackerState(
            transform_World_Camera,
            transform_World_Object,
            condor_state_0,
            warp_gmm_state,
            object_idx,
            True,
        ),
        {},  # debug data
    )


def update(
    state: ObjectTrackerState,
    frame: Frame,
    timestep: int,
    hypers=DEFAULT_HYPERS,
    debug=False,
    do_matter_update_override=None,
):
    warp_gmm_state = state.warp_gmm_state
    warp_gmm_state.gmm.object_posquats.requires_grad = True

    learning_rates = wp.array(
        [0.002, 0.002, 0.002, 0.0004, 0.0004, 0.0004, 0.0004], dtype=wp.float32
    )

    frame_warp = frame.as_warp()
    _ = warp_gmm.optimize_params(
        [warp_gmm_state.gmm.object_posquats],
        frame_warp,
        warp_gmm_state,
        num_timesteps=200,
        lr=learning_rates,
        storing_stuff=debug,
    )
    transform_World_Camera = Pose(wp.to_jax(warp_gmm_state.gmm.camera_posquat))
    transform_World_Object = Pose(wp.to_jax(warp_gmm_state.gmm.object_posquats)[0])

    if hypers.use_gt_camera_pose:
        # I added this branch to override the camera pose with the ground truth
        # in order to make it easier to isolate object tracking issues.
        transform_World_Camera = Pose(jnp.array(frame.camera_pose))
        # TODO: ensure warp code compatibility with this toggle.

    if do_matter_update_override is None:
        do_matter_update = (
            hypers.matter_update_period > 0
            and timestep % hypers.matter_update_period == 0
        )
    else:
        do_matter_update = do_matter_update_override

    if do_matter_update:
        (bkg_gmm_Camera_frame, object_gmm_object_frame, condor_state) = condor_update(
            frame,
            transform_World_Camera,
            transform_World_Object,
            state.condor_state,
            hypers.get_condor_config(),
        )
        # bkg_gmm_world_frame = bkg_gmm_Camera_frame.transform_by(transform_World_Camera)  # noqa
        warp_gmm_object = warp_gmm.gmm_warp_from_gmm_jax(object_gmm_object_frame)
        warp_gmm_object.camera_posquat = wp.from_jax(transform_World_Camera.posquat)
        warp_gmm_object.object_posquats = wp.array(
            transform_World_Object.posquat[None, :], dtype=wp.float32
        )
        warp_gmm_state.gmm = warp_gmm_object
    else:
        condor_state = state.condor_state

    return (
        transform_World_Camera,
        transform_World_Object,
        ObjectTrackerState(
            transform_World_Camera,
            transform_World_Object,
            condor_state,
            warp_gmm_state,
            state.object_idx,
            do_matter_update,
        ),
        {},  # debug data
    )


def rr_log(
    state: ObjectTrackerState,
    frame: Frame,
    timestep: int,
    hypers=DEFAULT_HYPERS,
    do_log_condor_state=True,
    do_log_frame=True,
    do_log_poses=True,
    log_inferred_camera_frustum=True,
):
    if do_log_condor_state or do_log_frame or do_log_poses:
        rr_set_time(timestep)
    if state.did_matter_update and do_log_condor_state:
        log_condor_state(
            state.condor_state.state, state.condor_state.hypers, log_in_world_frame=True
        )
    if do_log_poses:
        rr_log_pose(state.camera_pose, "camera_pose/inferred")
        rr_log_pose(Pose(frame.camera_pose), "camera_pose/ground_truth")
        rr_log_pose(state.object_pose, "object_pose/inferred")
        rr_log_pose(
            Pose(frame.object_poses[state.object_idx]), "object_pose/ground_truth"
        )
    if log_inferred_camera_frustum:
        rr_log_frustum(
            state.camera_pose,
            frame.intrinsics[0],
            frame.intrinsics[1],
            frame.depth.shape[0],
            frame.depth.shape[1],
            channel="inferred_camera_frustum",
        )

    if do_log_frame:
        rr_log_frame(frame, "observed_data")
        rr_log_frame(
            frame,
            "observed_data_unprojected_from_inferred_camera_frame",
            camera_pose=state.camera_pose,
        )

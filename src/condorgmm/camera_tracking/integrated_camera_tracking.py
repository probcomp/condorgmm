import jax
from dataclasses import dataclass, field
from condorgmm.data import Frame
from condorgmm import rr_set_time
from condorgmm import Pose
import condorgmm.warp_gmm as warp_gmm
import jax.numpy as jnp
import condorgmm
import condorgmm.utils.common.rerun
import warp as wp
import rerun as rr

from condorgmm.condor.interface.camera_tracking import (
    CondorCameraTrackerState,
    fast_config as condor_fast_config,
    initialize as condor_initialize,
    update as condor_update,
)
from condorgmm.condor.rerun import log_state as log_condor_state


@dataclass
class CameraTrackerHyperparams:
    matter_update_period: int
    n_gaussians: int
    tile_size_x: int = 8
    tile_size_y: int = 8
    condor_repopulate_depth_nonreturns: bool = False
    do_condor_pose_update: bool = False
    crop_fraction_x: float = 0.6
    crop_fraction_y: float = 0.6

    def get_condor_config(self):
        return condor_fast_config.replace(
            n_gaussians=self.n_gaussians,
            tile_size_x=self.tile_size_x,
            tile_size_y=self.tile_size_y,
            repopulate_depth_nonreturns=self.condor_repopulate_depth_nonreturns,
            do_pose_update=self.do_condor_pose_update,
        )


DEFAULT_HYPERS = CameraTrackerHyperparams(1, 600)


@dataclass
class CameraTrackerState:
    camera_pose: Pose  # uses jax arrays
    condor_state: CondorCameraTrackerState
    warp_gmm_state: warp_gmm.State
    did_matter_update: bool
    inferred_camera_poses: dict = field(default_factory=lambda: {})


def initialize(frame: Frame, debug=False, hypers=DEFAULT_HYPERS, seed=0, em=False):
    wp.init()
    transform_World_Camera = condorgmm.Pose(jnp.array(frame.camera_pose))
    (gmm0_camera_frame, condor_state_0) = condor_initialize(
        frame,
        transform_World_Camera,
        hypers.get_condor_config(),
        key=jax.random.key(seed),
    )
    gmm0_world_frame = gmm0_camera_frame.transform_by(transform_World_Camera)
    gmm_warp_world_frame = warp_gmm.gmm_warp_from_gmm_jax(gmm0_world_frame)
    warpgmm_state = warp_gmm.initialize_state(gmm=gmm_warp_world_frame, frame=frame)
    warpgmm_state.hyperparams.window_half_width = 8

    inferred_camera_poses = {}
    inferred_camera_poses[0] = transform_World_Camera.posquat
    inferred_camera_poses[-1] = transform_World_Camera.posquat

    warpgmm_state.gmm.camera_posquat = wp.from_jax(transform_World_Camera.posquat)
    warpgmm_state.gmm.camera_posquat.requires_grad = True

    return (
        transform_World_Camera,
        CameraTrackerState(
            transform_World_Camera,
            condor_state_0,
            warpgmm_state,
            True,
            inferred_camera_poses,
        ),
        {},
    )


def update(
    state: CameraTrackerState,
    frame: Frame,
    timestep,
    debug=False,
    hypers=DEFAULT_HYPERS,
    do_matter_update_override=None,
):
    warp_gmm_state = state.warp_gmm_state

    prev_prev_pose = state.inferred_camera_poses[timestep - 2]
    prev_pose = state.inferred_camera_poses[timestep - 1]
    prev_prev_quat = prev_prev_pose[3:]
    prev_quat = prev_pose[3:]

    prev_prev_quat = prev_prev_quat / jnp.linalg.norm(prev_prev_quat)
    prev_quat = prev_quat / jnp.linalg.norm(prev_quat)

    new_quat = prev_quat + (prev_quat - prev_prev_quat)
    new_quat = new_quat / jnp.linalg.norm(new_quat)

    new_pose = jnp.array(
        [*(prev_pose[:3] + (prev_pose[:3] - prev_prev_pose[:3])), *(new_quat)]
    )

    warp_gmm_state.gmm.camera_posquat = wp.from_jax(new_pose)
    warp_gmm_state.gmm.camera_posquat.requires_grad = True

    learning_rates = wp.array(
        [0.002, 0.002, 0.002, 0.0004, 0.0004, 0.0004, 0.0004], dtype=wp.float32
    )

    frame_warp = frame.crop_to_fraction(
        hypers.crop_fraction_x, hypers.crop_fraction_y
    ).as_warp()
    optimization_data = warp_gmm.optimize_params(
        [warp_gmm_state.gmm.camera_posquat],
        frame_warp,
        warp_gmm_state,
        num_timesteps=500,
        lr=learning_rates,
        storing_stuff=debug,
    )
    transform_World_Camera = Pose(wp.to_jax(warp_gmm_state.gmm.camera_posquat))

    if do_matter_update_override is None:
        do_matter_update = (
            hypers.matter_update_period > 0
            and timestep % hypers.matter_update_period == 0
        )
    else:
        do_matter_update = do_matter_update_override

    if do_matter_update:
        gmm_Camera_frame, condor_state = condor_update(
            frame,
            transform_World_Camera,
            state.condor_state,
            hypers.get_condor_config(),
        )
        transform_World_Camera = Pose(
            condor_state.state.scene.transform_World_Camera.posquat
        )
        gmm_world_frame = gmm_Camera_frame.transform_by(transform_World_Camera)
        gmm_warp_world_frame = warp_gmm.gmm_warp_from_gmm_jax(gmm_world_frame)
        gmm_warp_world_frame.camera_posquat = wp.from_jax(
            transform_World_Camera.posquat
        )
        warp_gmm_state.gmm = gmm_warp_world_frame
    else:
        condor_state = state.condor_state

    inferred_camera_poses = state.inferred_camera_poses
    inferred_camera_poses[timestep] = transform_World_Camera.posquat

    return (
        transform_World_Camera,
        CameraTrackerState(
            transform_World_Camera,
            condor_state,
            warp_gmm_state,
            do_matter_update,
            inferred_camera_poses,
        ),
        {
            "optimization_data": optimization_data,
            "gmm_is_valid": warp_gmm_state.gmm.is_valid(),
        },
    )


def rr_log(
    state: CameraTrackerState,
    frame: Frame,
    timestep: int,
    hypers=DEFAULT_HYPERS,
    do_log_condor_state=True,
    do_log_poses=True,
    do_log_frame=True,
    log_frames_to_next_timestep=True,
    log_inferred_camera_frustum=True,
    ellipse_mode=rr.components.FillMode.MajorWireframe,
    ellipse_scalar=1,
):
    rr_set_time(timestep)
    if state.did_matter_update and do_log_condor_state:
        log_condor_state(
            state.condor_state.state,
            state.condor_state.hypers,
            log_in_world_frame=True,
            ellipse_mode=ellipse_mode,
            ellipse_scalar=ellipse_scalar,
        )
    if do_log_poses:
        condorgmm.utils.common.rerun.rr_log_pose(state.camera_pose, "inferred_camera_pose")
        condorgmm.utils.common.rerun.rr_log_pose(
            Pose(frame.camera_pose), "true_camera_pose"
        )

    if log_inferred_camera_frustum:
        condorgmm.utils.common.rerun.rr_log_frustum(
            state.camera_pose,
            frame.intrinsics[0],
            frame.intrinsics[1],
            frame.depth.shape[0],
            frame.depth.shape[1],
            channel="inferred_camera_frustum",
        )

    if do_log_frame:
        condorgmm.utils.common.rerun.rr_log_frame(
            frame, "observed_data", camera_pose=condorgmm.Pose(frame.camera_pose)
        )
        condorgmm.utils.common.rerun.rr_log_frame(
            frame,
            "observed_data_unprojected_from_inferred_camera_frame",
            camera_pose=state.camera_pose,
        )

    if log_frames_to_next_timestep:
        rr_set_time(timestep + 1)
        condorgmm.utils.common.rerun.rr_log_frame(
            frame, "prev_frame/observed_data", camera_pose=condorgmm.Pose(frame.camera_pose)
        )
        condorgmm.utils.common.rerun.rr_log_frame(
            frame,
            "prev_frame/observed_data_unprojected_from_inferred_camera_frame",
            camera_pose=state.camera_pose,
        )
        rr_set_time(timestep)

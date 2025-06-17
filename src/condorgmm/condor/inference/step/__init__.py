import jax
import jax.numpy as jnp
from genjax import Mask
from functools import partial
from ...pose import Pose
from ...types import (
    Hyperparams,
    Observation,
    CondorGMMState,
    SceneState,
    Datapoint,
    BackgroundOnlySceneState,
    SingleKnownObjectSceneState,
)
import condorgmm.condor.model.model as model
from ..instrumentation import (
    LogConfig,
    sequence,
    simplemeta,
    default_config,
    prepend_label,
    Metadata,
)
from .gibbs import run_n_mcmc_sweeps, update_datapoint_associations_and_depths
from ..shared import reinitialize_unobserved_gaussians


@partial(
    jax.jit,
    static_argnames=("n_sweeps_first_pass", "n_sweeps_second_pass", "run_second_pass"),
)
def run_inference(
    key,
    prev_st: CondorGMMState,
    observations: Observation,
    hypers: Hyperparams,
    *,
    transform_World_Camera: Pose,
    transform_World_Object: Pose | None = None,
    n_sweeps_first_pass: int,
    n_sweeps_second_pass: int,
    run_second_pass: bool,
    c: LogConfig = default_config,
) -> tuple[CondorGMMState, Metadata]:
    print("tracing condor.inference.step.run_inference")
    # first pass
    pass1_st0 = _phase1_initialization(
        key,
        prev_st,
        observations,
        hypers,
        transform_World_Camera,
        transform_World_Object,
    )
    pass1_st1, pass1_meta1 = update_datapoint_associations_and_depths(
        key,
        pass1_st0,
        hypers.replace(always_accept_assoc_depth_move=True),
        c,
    )
    pass1_st2, pass1_meta2 = run_n_mcmc_sweeps(
        key, pass1_st1, prev_st, hypers, n_sweeps_first_pass, c
    )
    meta_first_pass = sequence(pass1_meta1, pass1_meta2)
    if not run_second_pass:
        return pass1_st2, meta_first_pass

    # second pass
    pass2_st0 = _phase2_initialization(key, pass1_st2, hypers)
    pass2_st1, pass2_meta1 = update_datapoint_associations_and_depths(
        key,
        pass2_st0,
        hypers.replace(always_accept_assoc_depth_move=True),
        c,
    )
    pass2_st2, pass2_meta2 = run_n_mcmc_sweeps(
        key, pass2_st1, prev_st, hypers, n_sweeps_second_pass, c
    )
    meta_second_pass = sequence(
        simplemeta(pass2_st0, c, "phase2::initialization", {}),
        prepend_label(pass2_meta1, "phase2::depth_assoc_update::"),
        prepend_label(pass2_meta2, "phase2::"),
    )

    # return
    return pass2_st2, sequence(meta_first_pass, meta_second_pass)


def _phase1_initialization(
    key,
    prev_st: CondorGMMState,
    observations: Observation,
    hypers: Hyperparams,
    transform_World_Camera: Pose,
    transform_World_Object: Pose | None,
) -> CondorGMMState:
    new_scene = _update_scene_state_with_poses(
        prev_st.scene, transform_World_Camera, transform_World_Object
    )
    p: model.NIWParams = jax.vmap(
        lambda g: model.get_evolve_xyz_prior_params(
            g,
            new_scene,
            prev_st.scene,
            prev_st.matter.background_evolution_params,
            hypers,
        )
    )(prev_st.gaussians)
    new_gaussians = prev_st.gaussians.replace(
        xyz=p.prior_mean,
        xyz_cov=p.prior_cov,
        origin=jnp.arange(len(prev_st.gaussians)),
        n_frames_since_last_had_assoc=jax.vmap(
            lambda n, had_assoc: model.update_n_frames_since_last_assoc(n, had_assoc)
        )(
            prev_st.gaussians.n_frames_since_last_had_assoc,
            prev_st.gaussian_has_assoc_mask,
        ),
    )

    if hypers.repopulate_depth_nonreturns:
        datapoint_mask = hypers.datapoint_mask
    else:
        datapoint_mask = jnp.logical_and(
            hypers.datapoint_mask, observations.depth != 0.0
        )

    return prev_st.replace(
        scene=new_scene,
        matter=prev_st.matter.replace(
            gaussians=new_gaussians,
            tiling=prev_st.matter.tiling.update_tiling(new_gaussians, key),
        ),
        datapoints=Mask(
            jax.vmap(
                lambda obs: Datapoint.from_obs_det(
                    obs=obs, gaussian_idx=jnp.array(-1, dtype=int), hypers=hypers
                )
            )(observations),
            datapoint_mask,
        ),
    )


def _phase2_initialization(
    key,
    st: CondorGMMState,
    hypers: Hyperparams,
) -> CondorGMMState:
    return reinitialize_unobserved_gaussians(key, st, hypers)


def _update_scene_state_with_poses(
    scene: SceneState,
    transform_World_Camera: Pose,
    transform_World_Object: Pose | None,
):
    if isinstance(scene, BackgroundOnlySceneState):
        return scene.replace(transform_World_Camera=transform_World_Camera)
    elif isinstance(scene, SingleKnownObjectSceneState):
        assert transform_World_Object is not None
        return scene.replace(
            transform_World_Camera=transform_World_Camera,
            transform_World_Object=transform_World_Object,
        )
    raise ValueError(f"Unknown scene type: {scene}")

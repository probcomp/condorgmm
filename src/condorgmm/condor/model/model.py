import jax
import jax.numpy as jnp
import genjax
from genjax import ChoiceMapBuilder as C
from genjax import gen, Mask
from ..pose import Pose, gaussian_vmf
from ..geometry import (
    xyz_to_cameraxyd,
    isovars_and_quaternion_to_cov,
    cov_to_isovars_and_quaternion,
)
from ..types import (
    Hyperparams,
    Gaussian,
    NewGaussianPriorParams,
    VisualMatter,
    SceneState,
    Observation,
    Datapoint,
    CondorGMMState,
    Tiling,
    BackgroundOnlySceneState,
    SingleKnownObjectSceneState,
    GAMMA_RATE_PARAMETER,
    NIWParams,
    NIGParams,
    BackgroundGaussianEvolutionParams,
    FloatFromDiscreteSet,
)
from ..tiling import MonolithicTiling
from .distributions import (
    gamma,
    my_inverse_gamma,
    my_inverse_wishart,
    uniform_from_domain,
)
from genjax.typing import IntArray, BoolArray
from typing import cast

### Initial model ###


@gen
def init_model(hypers: Hyperparams) -> CondorGMMState:
    scene = generate_initial_scene(hypers) @ "scene"
    matter = generate_initial_matter(scene, hypers) @ "matter"
    datapoints = generate_datapoints(matter, hypers) @ "datapoints"
    return CondorGMMState(scene, matter, datapoints)


@gen
def generate_initial_scene(
    hypers: Hyperparams,
) -> BackgroundOnlySceneState | SingleKnownObjectSceneState:
    return hypers.initial_scene


@gen
def generate_initial_matter(scene: SceneState, hypers: Hyperparams) -> VisualMatter:
    params = new_gaussian_params_hyperprior(hypers) @ "global_params"
    gaussians = (
        generate_new_gaussian.vmap(in_axes=(0, None, None, None, None))(
            jnp.arange(hypers.n_gaussians), params, scene, hypers, True
        )
        @ "gaussians"
    )
    tiling = generate_tiling(gaussians, hypers) @ "tiling"
    return VisualMatter(
        params, hypers.default_background_evolution_params, gaussians, tiling
    )


@gen
def new_gaussian_params_hyperprior(
    hypers: Hyperparams,
) -> NewGaussianPriorParams[FloatFromDiscreteSet]:
    doms = hypers.prior_param_domains
    xyz_cov_pcnt = uniform_from_domain(doms.cov_pseudocount_domain_3d) @ "xyz_cov_pcnt"
    xyz_cov_isotropic_prior_stds = (
        uniform_from_domain.repeat(n=3)(doms.xyz_pseudo_std_domain)
        @ "xyz_cov_isotropic_prior_stds"
    )
    xyz_mean_pcnt = uniform_from_domain(doms.mean_pseudocount_domain) @ "xyz_mean_pcnt"

    rgb_var_n_pseudo_obs = (
        uniform_from_domain.repeat(n=3)(doms.std_pseudocount_domain_1d)
        @ "rgb_var_n_pseudo_obs"
    )
    rgb_var_pseudo_sample_stds = (
        uniform_from_domain.repeat(n=3)(doms.rgb_pseudo_std_domain)
        @ "rgb_var_pseudo_sample_stds"
    )
    rgb_mean_n_pseudo_obs = (
        uniform_from_domain.repeat(n=3)(doms.mean_pseudocount_domain)
        @ "rgb_mean_n_pseudo_obs"
    )

    return NewGaussianPriorParams(
        xyz_cov_pcnt,
        xyz_cov_isotropic_prior_stds,
        xyz_mean_pcnt,
        rgb_var_n_pseudo_obs,
        rgb_var_pseudo_sample_stds,
        rgb_mean_n_pseudo_obs,
    )


@gen
def evolved_gaussian_params_hyperprior(
    hypers: Hyperparams,
) -> BackgroundGaussianEvolutionParams[FloatFromDiscreteSet]:
    doms = hypers.evolved_gaussian_prior_param_domains
    prob_gaussian_is_new = (
        uniform_from_domain(doms.prob_gaussian_is_new_domain) @ "prob_gaussian_is_new"
    )
    xyz_cov_pcnt = uniform_from_domain(doms.xyz_cov_pcnt_domain) @ "xyz_cov_pcnt"
    rgb_var_pcnt = uniform_from_domain(doms.rgb_var_pcnt_domain) @ "rgb_var_pcnt"
    target_xyz_mean_std = (
        uniform_from_domain(doms.target_xyz_mean_std_domain) @ "target_xyz_mean_std"
    )

    return BackgroundGaussianEvolutionParams(
        prob_gaussian_is_new=prob_gaussian_is_new,
        xyz_cov_pcnt=xyz_cov_pcnt,
        rgb_var_pcnt=rgb_var_pcnt,
        target_xyz_mean_std=target_xyz_mean_std,
    )


### Step model ###


@gen
def step_model(
    prev: CondorGMMState,
    hypers: Hyperparams,
) -> CondorGMMState:
    scene = evolve_scene(prev.scene, hypers) @ "scene"
    matter = evolve_matter(prev, scene, hypers) @ "matter"
    datapoints = generate_datapoints(matter, hypers) @ "datapoints"
    return CondorGMMState(scene=scene, matter=matter, datapoints=datapoints)


@gen
def _evolve_background_only_scene_state(
    prev: BackgroundOnlySceneState, hypers: Hyperparams
) -> BackgroundOnlySceneState:
    transform_World_Camera = (
        gaussian_vmf(
            prev.transform_World_Camera,
            hypers.camera_pose_drift_std,
            hypers.camera_pose_drift_concentration,
        )
        @ "transform_World_Camera"
    )
    return BackgroundOnlySceneState(
        transform_World_Camera=transform_World_Camera,
    )


@gen
def _evolve_single_known_object_scene_state(
    prev: SingleKnownObjectSceneState, hypers: Hyperparams
) -> SingleKnownObjectSceneState:
    transform_World_Camera = (
        gaussian_vmf(
            prev.transform_World_Camera,
            hypers.camera_pose_drift_std,
            hypers.camera_pose_drift_concentration,
        )
        @ "transform_World_Camera"
    )
    transform_World_Object = (
        gaussian_vmf(
            prev.transform_World_Object,
            hypers.object_pose_drift_std,
            hypers.object_pose_drift_concentration,
        )
        @ "transform_World_Object"
    )
    return SingleKnownObjectSceneState(
        transform_World_Camera=transform_World_Camera,
        transform_World_Object=transform_World_Object,
        object_model=prev.object_model,
    )


@gen
def evolve_scene(
    prev: SceneState, hypers: Hyperparams
) -> BackgroundOnlySceneState | SingleKnownObjectSceneState:
    if isinstance(prev, BackgroundOnlySceneState):
        return _evolve_background_only_scene_state.inline(prev, hypers)
    elif isinstance(prev, SingleKnownObjectSceneState):
        return _evolve_single_known_object_scene_state.inline(prev, hypers)
    else:
        raise ValueError(f"Unknown scene state type: {type(prev)}.")


@gen
def generate_origin(
    gaussian_idx: IntArray,
    params: BackgroundGaussianEvolutionParams[FloatFromDiscreteSet],
    hypers: Hyperparams,
) -> jnp.ndarray:
    prob_is_new = jnp.where(
        gaussian_is_background(gaussian_idx, hypers),
        params.values.prob_gaussian_is_new,
        jnp.array(0.0),
    )
    is_new = genjax.flip(prob_is_new) @ "is_new"
    return jnp.where(is_new, jnp.array(-1, dtype=jnp.int32), gaussian_idx)


def _get_alpha_for_evolve_object_gaussian_weight(
    prev: Gaussian, probs: jnp.ndarray, hypers: Hyperparams
):
    if not isinstance(hypers.initial_scene, SingleKnownObjectSceneState):
        return jnp.array(-1.0, dtype=float)

    do_reset = jnp.logical_or(
        prev.n_frames_since_last_had_assoc == -1,
        prev.n_frames_since_last_had_assoc
        > hypers.n_unobserved_frames_to_object_gaussian_reset,
    )
    object_model_gaussian = get_object_model_gaussian(prev.idx, hypers)
    prob_in_object_model = (
        object_model_gaussian.mixture_weight
        / hypers.initial_scene.object_model.mixture_weight.sum()
    )
    total_object_gaussian_prob = probs[len(hypers.initial_scene.object_model) :].sum()
    prob_in_model_overall = total_object_gaussian_prob * prob_in_object_model
    return jnp.where(
        do_reset,
        (
            jnp.maximum(prob_in_model_overall, probs[prev.idx])
            * hypers.alpha_multiplier_for_evolved_gaussian
        ),
        probs[prev.idx] * hypers.alpha_multiplier_for_evolved_gaussian,
    )


def get_alpha_for_evolve_gaussian_weight(
    prev: Gaussian, probs: jnp.ndarray, hypers: Hyperparams
):
    val_if_bkg = probs[prev.origin] * hypers.alpha_multiplier_for_evolved_gaussian
    val_if_obj = _get_alpha_for_evolve_object_gaussian_weight(prev, probs, hypers)
    return jnp.where(gaussian_is_background(prev.idx, hypers), val_if_bkg, val_if_obj)


def _next_bkg_gaussian_pose_under_rigid_motion(
    prev_pose: Pose,
    prev_scene: BackgroundOnlySceneState | SingleKnownObjectSceneState,
    scene: BackgroundOnlySceneState | SingleKnownObjectSceneState,
) -> Pose:
    transform_prevWorld_prevCamera = prev_scene.transform_World_Camera
    transform_newCamera_NewWorld = scene.transform_World_Camera.inv()
    transform_prevCamera_prevGaussian = prev_pose
    transform_prevWorld_prevGaussian = (
        transform_prevWorld_prevCamera @ transform_prevCamera_prevGaussian
    )
    transform_newWorld_newGaussian = transform_prevWorld_prevGaussian
    transform_newCamera_newGaussian = (
        transform_newCamera_NewWorld @ transform_newWorld_newGaussian
    )
    return transform_newCamera_newGaussian


def _get_evolve_xyz_prior_params_if_background(
    prev: Gaussian,
    scene: BackgroundOnlySceneState | SingleKnownObjectSceneState,
    prev_scene: BackgroundOnlySceneState | SingleKnownObjectSceneState,
    params: BackgroundGaussianEvolutionParams[FloatFromDiscreteSet],
    hypers: Hyperparams,
) -> NIWParams:
    isovars, quat = cov_to_isovars_and_quaternion(prev.xyz_cov)
    predicted_pose = _next_bkg_gaussian_pose_under_rigid_motion(
        Pose(prev.xyz, quat), prev_scene, scene
    )
    predicted_cov = isovars_and_quaternion_to_cov(isovars, predicted_pose.quaternion)
    cov_pcnt = params.values.xyz_cov_pcnt
    # TODO: I'm not sure exactly the best form of this pcnt, so it
    # may be worth playing with this more.
    mean_pcnt = jnp.sort(isovars)[1] / (params.values.target_xyz_mean_std**2)
    return NIWParams(
        cov_pcnt=cov_pcnt,
        prior_cov=predicted_cov,
        mean_pcnt=mean_pcnt,
        prior_mean=predicted_pose.pos,
    )


def _get_evolve_xyz_prior_params_if_object(
    prev: Gaussian,
    scene: SingleKnownObjectSceneState,
    prev_scene: SingleKnownObjectSceneState,
    hypers: Hyperparams,
) -> NIWParams:
    target = get_object_model_gaussian_in_camera_frame(prev.idx, scene, hypers)
    return NIWParams(
        cov_pcnt=hypers.xyz_cov_evolution_pcnt_object,
        mean_pcnt=hypers.xyz_mean_evolution_pcnt_object,
        prior_mean=target.xyz,
        prior_cov=target.xyz_cov,
    )


def get_evolve_xyz_prior_params(
    prev: Gaussian,
    scene: BackgroundOnlySceneState | SingleKnownObjectSceneState,
    prev_scene: BackgroundOnlySceneState | SingleKnownObjectSceneState,
    params: BackgroundGaussianEvolutionParams[FloatFromDiscreteSet],
    hypers: Hyperparams,
) -> NIWParams:
    params_if_background = _get_evolve_xyz_prior_params_if_background(
        prev, scene, prev_scene, params, hypers
    )
    if isinstance(scene, BackgroundOnlySceneState):
        return params_if_background

    prev_scene = cast(SingleKnownObjectSceneState, prev_scene)
    params_if_object = _get_evolve_xyz_prior_params_if_object(
        prev, scene, prev_scene, hypers
    )
    return jax.tree_map(
        lambda x, y: jnp.where(gaussian_is_background(prev.idx, hypers), x, y),
        params_if_background,
        params_if_object,
    )


def _get_evolve_rgb_prior_params_if_background(
    prev: Gaussian,
    params: BackgroundGaussianEvolutionParams[FloatFromDiscreteSet],
    hypers: Hyperparams,
) -> NIGParams:
    rgb_mean_center = prev.rgb
    rgb_prior_vars = prev.rgb_vars
    rgb_pcnts = params.values.rgb_var_pcnt * jnp.ones(3, dtype=jnp.float32)
    rgb_mean_pcnts = (
        rgb_prior_vars / hypers.target_rgb_mean_variance_for_object_evolution
    )
    return NIGParams(
        var_pcnt=rgb_pcnts,
        prior_var=rgb_prior_vars,
        mean_pcnt=rgb_mean_pcnts,
        prior_mean=rgb_mean_center,
    )


def _get_object_gaussian_evolve_rgb_targets(prev: Gaussian, hypers: Hyperparams):
    model_gaussian = get_object_model_gaussian(prev.idx, hypers)
    model_rgb = model_gaussian.rgb
    model_vars = model_gaussian.rgb_vars
    prev_rgb = prev.rgb
    prev_vars = prev.rgb_vars

    do_reset = jnp.logical_or(
        prev.n_frames_since_last_had_assoc == -1,
        prev.n_frames_since_last_had_assoc
        > hypers.n_unobserved_frames_to_object_gaussian_reset,
    )
    target_rgb_mean = jnp.where(do_reset, model_rgb, prev_rgb)
    target_variance_for_rgb_mean = jnp.where(
        do_reset,
        jnp.ones(3, dtype=float)
        * hypers.target_rgb_mean_variance_object_initialization,
        jnp.ones(3, dtype=float) * hypers.target_rgb_mean_variance_for_object_evolution,
        #                          ^ have this be lower so unobserved Gaussians retain their previous color
    )
    target_rgb_vars = jnp.where(do_reset, model_vars, prev_vars)
    pcnt_for_rgb_vars = jnp.where(
        do_reset,
        hypers.rgb_var_pcnt_object_initialization,
        hypers.rgb_var_evolution_pcnt_object,
    )
    return (
        target_rgb_mean,
        target_variance_for_rgb_mean,
        target_rgb_vars,
        pcnt_for_rgb_vars,
    )


def _get_evolve_rgb_prior_params_if_object(
    prev: Gaussian, hypers: Hyperparams
) -> NIGParams:
    (
        target_rgb_mean,
        target_variance_for_rgb_mean,
        target_rgb_vars,
        pcnt_for_rgb_vars,
    ) = _get_object_gaussian_evolve_rgb_targets(prev, hypers)

    return NIGParams(
        prior_mean=target_rgb_mean,
        mean_pcnt=target_rgb_vars / target_variance_for_rgb_mean,
        var_pcnt=pcnt_for_rgb_vars,
        prior_var=target_rgb_vars,
    )


def get_evolve_rgb_prior_params(
    prev: Gaussian,
    params: BackgroundGaussianEvolutionParams[FloatFromDiscreteSet],
    hypers: Hyperparams,
) -> NIGParams:
    params_if_background = _get_evolve_rgb_prior_params_if_background(
        prev, params, hypers
    )
    if isinstance(hypers.initial_scene, BackgroundOnlySceneState):
        return params_if_background

    params_if_object = _get_evolve_rgb_prior_params_if_object(prev, hypers)
    return jax.tree_map(
        lambda x, y: jnp.where(gaussian_is_background(prev.idx, hypers), x, y),
        params_if_background,
        params_if_object,
    )


@gen
def evolve_gaussian(
    idx: IntArray,
    origin,
    prev: Gaussian,
    prev_probs: jnp.ndarray,
    had_assoc_at_prev_frame: BoolArray,
    scene: BackgroundOnlySceneState | SingleKnownObjectSceneState,
    prev_scene: BackgroundOnlySceneState | SingleKnownObjectSceneState,
    params: BackgroundGaussianEvolutionParams[FloatFromDiscreteSet],
    hypers: Hyperparams,
) -> Gaussian:
    xyzp = get_evolve_xyz_prior_params(prev, scene, prev_scene, params, hypers)
    xyz_cov = my_inverse_wishart(xyzp.cov_pcnt, xyzp.prior_cov) @ "xyz_cov"
    xyz = genjax.mv_normal(xyzp.prior_mean, xyz_cov / xyzp.mean_pcnt) @ "xyz"

    rgbp = get_evolve_rgb_prior_params(prev, params, hypers)
    rgb_vars = (
        my_inverse_gamma.vmap(in_axes=(0, 0))(rgbp.var_pcnt, rgbp.prior_var)
        @ "rgb_vars"
    )
    rgb_stds = jnp.sqrt(rgb_vars / rgbp.mean_pcnt)
    rgb = genjax.normal.vmap(in_axes=(0, 0))(rgbp.prior_mean, rgb_stds) @ "rgb"

    mixture_weight = (
        gamma(
            get_alpha_for_evolve_gaussian_weight(prev, prev_probs, hypers),
            GAMMA_RATE_PARAMETER,
        )
        @ "mixture_weight"
    )

    return Gaussian(
        idx=idx,
        xyz=xyz,
        xyz_cov=xyz_cov,
        rgb=rgb,
        rgb_vars=rgb_vars,
        mixture_weight=mixture_weight,
        origin=origin,
        object_idx=jnp.array(0, dtype=int),
        n_frames_since_last_had_assoc=update_n_frames_since_last_assoc(
            prev.n_frames_since_last_had_assoc, had_assoc_at_prev_frame
        ),
    )


def update_n_frames_since_last_assoc(
    n_frames_since_last_had_assoc: IntArray, had_assoc_at_prev_frame: BoolArray
):
    return jnp.where(
        had_assoc_at_prev_frame,
        jnp.array(1, dtype=int),
        jnp.where(
            n_frames_since_last_had_assoc == -1,
            -1,  # never had assoc before -> still has never had assoc
            1 + n_frames_since_last_had_assoc,
        ),
    )


@gen
def generate_gaussian_i_at_noninitial_timestep(
    idx: int,
    scene: SceneState,
    prev_scene: SceneState,
    prev_matter: VisualMatter,
    prev_had_assoc_mask: jnp.ndarray,
    hypers: Hyperparams,
) -> Gaussian:
    params = prev_matter.background_evolution_params
    origin = generate_origin(idx, params, hypers) @ "origin"
    gaussian = (
        genjax.switch(generate_new_gaussian, evolve_gaussian)(
            jnp.where(origin == -1, 0, 1),
            (idx, prev_matter.background_initialization_params, scene, hypers, False),
            (
                idx,
                origin,
                prev_matter.gaussians[origin],
                prev_matter.probs,
                prev_had_assoc_mask[origin],
                scene,
                prev_scene,
                params,
                hypers,
            ),
        )
        @ "gaussian"
    )
    return gaussian


@gen
def evolve_matter(
    prev_st: CondorGMMState, new_scene: SceneState, hypers: Hyperparams
) -> VisualMatter:
    evolution_params = evolved_gaussian_params_hyperprior(hypers) @ "evolution_params"
    gaussians = (
        generate_gaussian_i_at_noninitial_timestep.vmap(
            in_axes=(0, None, None, None, None, None)
        )(
            jnp.arange(hypers.n_gaussians),
            new_scene,
            prev_st.scene,
            prev_st.matter,
            prev_st.gaussian_has_assoc_mask,
            hypers,
        )
        @ "gaussians"
    )
    tiling = generate_tiling(gaussians, hypers) @ "tiling"
    return VisualMatter(
        prev_st.matter.background_initialization_params,
        evolution_params,
        gaussians,
        tiling,
    )


### Shared ###


def get_new_gaussian_xyz_params(idx, params, scene, hypers) -> NIWParams:
    world_pos_camera_frame = scene.transform_World_Camera.inv().pos
    params_if_background = params.xyz_params(world_pos_camera_frame)
    if isinstance(scene, BackgroundOnlySceneState):
        return params_if_background

    target = get_object_model_gaussian_in_camera_frame(idx, scene, hypers)
    params_if_object = NIWParams(
        cov_pcnt=hypers.xyz_cov_pcnt_object_initialization,
        mean_pcnt=hypers.xyz_mean_pcnt_object_initialization,
        prior_mean=target.xyz,
        prior_cov=target.xyz_cov,
    )
    return jax.tree.map(
        lambda x, y: jnp.where(gaussian_is_background(idx, hypers), x, y),
        params_if_background,
        params_if_object,
    )


def get_new_gaussian_rgb_params(idx, params, scene, hypers) -> NIGParams:
    params_if_background = params.rgb_params

    if isinstance(scene, BackgroundOnlySceneState):
        return params_if_background

    target = get_object_model_gaussian(idx, hypers)
    mean_pcnt = target.rgb_vars / hypers.target_rgb_mean_variance_object_initialization
    params_if_object = NIGParams(
        var_pcnt=hypers.rgb_var_pcnt_object_initialization,
        mean_pcnt=mean_pcnt,
        prior_mean=target.rgb,
        prior_var=target.rgb_vars,
    )
    return jax.tree.map(
        lambda x, y: jnp.where(gaussian_is_background(idx, hypers), x, y),
        params_if_background,
        params_if_object,
    )


def new_gaussian_mixture_weight_alpha(
    gaussian_idx: IntArray,
    is_frame_0: bool,
    hypers: Hyperparams,
):
    gamma_alpha_if_bkg = jnp.where(
        is_frame_0,
        hypers.initial_crp_alpha_background / hypers.n_gaussians,
        hypers.crp_alpha_for_new_background_gaussian_in_step_model / hypers.n_gaussians,
    )
    gamma_alpha_if_obj = hypers.initial_crp_alpha_object / hypers.n_gaussians
    return jnp.where(
        gaussian_is_background(gaussian_idx, hypers),
        gamma_alpha_if_bkg,
        gamma_alpha_if_obj,
    )


@gen
def generate_new_gaussian(
    idx: IntArray,
    params: NewGaussianPriorParams[FloatFromDiscreteSet],
    scene: SceneState,
    hypers: Hyperparams,
    is_frame_0: bool = False,
) -> Gaussian:
    # Under the notation of
    # https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution,
    # nu = cov_pcnt,
    # Psi = prior_cov
    # mu_0 = prior_mean,
    # lambda = mean_pcnt
    p1: NIWParams = get_new_gaussian_xyz_params(idx, params, scene, hypers)
    # This next line is needed to make it possible to simulate from the model,
    # due to https://github.com/tensorflow/probability/issues/1987.
    cov_pcnt = jnp.where(
        hypers.running_simulate, jnp.maximum(p1.cov_pcnt, 3.0), p1.cov_pcnt
    )

    xyz_cov = my_inverse_wishart(cov_pcnt, p1.prior_cov) @ "xyz_cov"
    xyz = genjax.mv_normal(p1.prior_mean, xyz_cov / p1.mean_pcnt) @ "xyz"

    # Under the notation of
    # https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution, we have
    # alpha = rgb_var_n_pseudo_obs[i] / 2
    # beta = rgb_var_pseudo_sample_vars[i] * alpha
    # mu = [255 / 2, 255 / 2, 255 / 2]
    # lambda = rgb_mean_n_pseudo_obs[i]
    # where `i` indexes into either the R, G, or B channel.
    p2: NIGParams = get_new_gaussian_rgb_params(idx, params, scene, hypers)
    rgb_vars = (
        my_inverse_gamma.vmap(in_axes=(0, 0))(p2.var_pcnt, p2.prior_var) @ "rgb_vars"
    )
    rgb_stds = jnp.sqrt(rgb_vars / p2.mean_pcnt)
    rgb = genjax.normal.vmap(in_axes=(0, 0))(p2.prior_mean, rgb_stds) @ "rgb"

    gamma_alpha = new_gaussian_mixture_weight_alpha(idx, is_frame_0, hypers)
    mixture_weight = gamma(gamma_alpha, GAMMA_RATE_PARAMETER) @ "mixture_weight"

    return Gaussian(
        idx=idx,
        xyz=xyz,
        xyz_cov=xyz_cov,
        rgb=rgb,
        rgb_vars=rgb_vars,
        mixture_weight=mixture_weight,
        origin=jnp.array(-1, dtype=int),
        object_idx=jnp.where(gaussian_is_background(idx, hypers), 0, 1),
        n_frames_since_last_had_assoc=jnp.array(-1, dtype=int),
    )


@gen
def generate_tiling(gaussians: jnp.ndarray, hypers: Hyperparams) -> Tiling:
    # Backlog TODO: switch to returning a high quality tiling.
    return MonolithicTiling(hypers.n_gaussians, len(hypers.datapoint_mask))


@gen
def generate_datapoints(matter: VisualMatter, hypers: Hyperparams) -> Mask[Datapoint]:
    n_datapoints = len(hypers.datapoint_mask)
    return (
        generate_datapoint.mask().vmap(in_axes=(0, 0, None, None))(
            hypers.datapoint_mask, jnp.arange(n_datapoints), matter, hypers
        )
        @ "datapoints"
    )


@gen
def generate_datapoint(
    idx: int, matter: VisualMatter, hypers: Hyperparams
) -> Datapoint:
    gaussian_idx = (
        genjax.categorical(
            jnp.where(
                jax.vmap(
                    lambda gaussian_idx: matter.tiling.gaussian_is_relevant_for_datapoint(
                        gaussian_idx, idx
                    )
                )(jnp.arange(hypers.n_gaussians)),
                jnp.log(matter.probs),
                -jnp.inf * jnp.ones(hypers.n_gaussians),
            )
        )
        @ "gaussian_idx"
    )
    gaussian = matter.gaussians[gaussian_idx]
    xyz = genjax.mv_normal(gaussian.xyz, gaussian.xyz_cov) @ "xyz"
    rgb = (
        genjax.normal.vmap(in_axes=(0, 0))(gaussian.rgb, jnp.sqrt(gaussian.rgb_vars))
        @ "rgb"
    )

    is_depth_nonreturn = genjax.flip(hypers.p_depth_nonreturn) @ "is_depth_nonreturn"
    camera_xyd = xyz_to_cameraxyd(xyz, hypers.intrinsics)
    camera_xy, d = camera_xyd[:2], camera_xyd[2]
    depth = jnp.where(is_depth_nonreturn, jnp.array(0, dtype=jnp.float32), d)
    return Datapoint(Observation(rgb, camera_xy, depth), xyz, gaussian_idx)


### Choicemap constructors ###


def origin_to_choicemap(origin):
    return C["is_new"].set(
        jnp.where(
            origin == -1,
            jnp.array(True, dtype=jnp.bool),
            jnp.array(False, dtype=jnp.bool),
        )
    )


def gaussian_to_choicemap_for_ggiant(gaussian) -> genjax.ChoiceMap:
    return C["origin"].set(origin_to_choicemap(gaussian.origin)) | C["gaussian"].set(
        gaussian_to_choicemap(gaussian)
    )


def gaussian_to_choicemap(gaussian: Gaussian):
    return C.kw(
        xyz=gaussian.xyz,
        xyz_cov=gaussian.xyz_cov,
        rgb=gaussian.rgb,
        rgb_vars=gaussian.rgb_vars,
        mixture_weight=gaussian.mixture_weight,
    )


### Helpers ###
def gaussian_is_background(gaussian_idx: IntArray, hypers: Hyperparams):
    return isinstance(
        hypers.initial_scene, BackgroundOnlySceneState
    ) or gaussian_idx < n_background_gaussians(hypers)


def is_bkg_mask(st: CondorGMMState, hypers: Hyperparams):
    return jax.vmap(lambda idx: gaussian_is_background(idx, hypers))(
        st.matter.gaussians.idx
    )


def n_background_gaussians(hypers: Hyperparams):
    if isinstance(hypers.initial_scene, BackgroundOnlySceneState):
        return hypers.n_gaussians
    elif isinstance(hypers.initial_scene, SingleKnownObjectSceneState):
        return hypers.n_gaussians - len(hypers.initial_scene.object_model)
    else:
        raise ValueError(f"Unknown scene state type: {type(hypers.initial_scene)}.")


def get_object_model_gaussian(idx: IntArray, hypers: Hyperparams) -> Gaussian:
    if not isinstance(hypers.initial_scene, SingleKnownObjectSceneState):
        raise ValueError("Expected initial scene to be SingleKnownObjectSceneState.")
    object_model = hypers.initial_scene.object_model
    idx_in_model = idx - n_background_gaussians(hypers)
    idx_in_model = jnp.clip(idx_in_model, 0, len(object_model) - 1)
    return object_model[idx_in_model]


def get_object_model_gaussian_in_camera_frame(
    idx: IntArray, scene: SingleKnownObjectSceneState, hypers: Hyperparams
) -> Gaussian:
    gaussian_object_frame = get_object_model_gaussian(idx, hypers)
    transform_Camera_Object = (
        scene.transform_World_Camera.inv() @ scene.transform_World_Object
    )
    return gaussian_object_frame.transform_by(transform_Camera_Object)

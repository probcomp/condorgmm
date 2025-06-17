import jax
import jax.numpy as jnp
from jax.random import split
import genjax
from genjax import Mask
from typing import Callable
from ...types import (
    Hyperparams,
    CondorGMMState,
    Domain,
    FloatFromDiscreteSet,
    Gaussian,
)
from ...model.distributions import my_inverse_wishart, my_inverse_gamma
from ...geometry import cov_to_isovars_and_quaternion, find_aligning_pose
from ..instrumentation import (
    LogConfig,
    sequence,
    default_config,
    Metadata,
    wrap,
    flatten_metadata,
)
from ..conjugate_updates import (
    mixture_weights_to_categorical_update,
    normal_my_inverse_wishart_update,
    normal_my_inverse_gamma_update,
)
from ..shared import (
    update_datapoint_associations_and_depths,
    get_relevant_datapoints_for_gaussian,
    update_tiling,
)
import condorgmm.condor.model.model as model
import warnings
from genjax.typing import IntArray, FloatArray


def run_n_mcmc_sweeps(
    key,
    initial_new_st: CondorGMMState,
    prev_st: CondorGMMState,
    hypers: Hyperparams,
    n_mcmc_sweeps: int,
    c: LogConfig = default_config,
) -> tuple[CondorGMMState, Metadata]:
    def kernel(st: CondorGMMState, key):
        st, m = mcmc_sweep(key, st, prev_st, hypers, c)
        return st, m

    final_st, batched_meta = jax.lax.scan(
        kernel, initial_new_st, split(key, n_mcmc_sweeps)
    )
    return final_st, flatten_metadata(batched_meta)


def mcmc_sweep(
    key,
    st: CondorGMMState,
    prev_st: CondorGMMState,
    hypers: Hyperparams,
    c: LogConfig = default_config,
) -> tuple[CondorGMMState, Metadata]:
    k1, k2, k3, k4, k5, k6, k7 = split(key, 7)
    st, m1 = gibbs_on_mixture_weights(k1, st, prev_st, hypers, c)
    st, m2 = gibbs_on_gaussian_params(k2, st, prev_st, hypers, c)
    st, m3 = gibbs_on_origins(k3, st, prev_st, hypers, c)
    st, m4 = update_tiling(k4, st, hypers, c)
    st, m5 = update_datapoint_associations_and_depths(k5, st, hypers, c)
    st, m6 = gibbs_on_evolution_params(k6, st, prev_st, hypers, c)
    st, m7 = update_camera_pose(k7, st, prev_st, hypers, c)
    return st, sequence(m1, m2, m3, m4, m5, m6, m7)


def update_camera_pose(
    key,
    st: CondorGMMState,
    prev_st: CondorGMMState,
    hypers: Hyperparams,
    c: LogConfig = default_config,
):
    if not hypers.do_pose_update:
        return wrap(
            st,
            c,
            "[no-op] (would be camera pose update if hypers.update_camera_pose were True)",
        )

    if not isinstance(st.scene, model.BackgroundOnlySceneState):
        warnings.warn(
            "Not running camera poes updates in step.gibbs since the scene is not a BackgroundOnlySceneState."
        )
        return wrap(st, c, "update_camera_pose")

    mask = jnp.logical_and(st.gaussians.origin >= 0, st.gaussian_has_assoc_mask)
    xyz_prev_camera_frame = prev_st.gaussians.xyz
    xyz_current_camera_frame = st.gaussians.xyz
    transform_currentframe_prevframe = find_aligning_pose(
        xyz_prev_camera_frame, xyz_current_camera_frame, mask
    )
    transform_world_prevframe = prev_st.scene.transform_World_Camera
    transform_world_current_camera_frame = (
        transform_world_prevframe @ transform_currentframe_prevframe.inv()
    )

    new_st = st.replace(
        {"scene": {"transform_World_Camera": transform_world_current_camera_frame}}
    )
    return wrap(new_st, c, "update_camera_pose")


def gibbs_on_mixture_weights(
    key,
    st: CondorGMMState,
    prev_st: CondorGMMState,
    hypers: Hyperparams,
    c: LogConfig = default_config,
):
    def get_alpha(gaussian):
        return jnp.where(
            gaussian.origin == -1,
            model.new_gaussian_mixture_weight_alpha(gaussian.idx, False, hypers),
            model.get_alpha_for_evolve_gaussian_weight(
                gaussian, prev_st.matter.probs, hypers
            ),
        )

    weights = mixture_weights_to_categorical_update(
        key,
        st.datapoints.value.gaussian_idx,
        st.datapoints.flag,
        jax.vmap(get_alpha)(st.matter.gaussians),
    )
    new_st = st.replace({"matter": {"gaussians": {"mixture_weight": weights}}})
    return wrap(new_st, c, "gibbs_on_mixture_weights")


def gibbs_on_gaussian_params(
    key,
    st: CondorGMMState,
    prev_st: CondorGMMState,
    hypers: Hyperparams,
    c: LogConfig = default_config,
):
    new_gaussians, meta_dict = jax.vmap(
        lambda key, idx: gibbs_on_one_gaussian(key, idx, st, prev_st, hypers, c)
    )(split(key, len(st.matter.gaussians)), jnp.arange(len(st.matter.gaussians)))
    new_st = st.replace({"matter": {"gaussians": new_gaussians}})
    return wrap(new_st, c, "gibbs_on_gaussian_params", meta_dict)


def gibbs_on_one_gaussian(
    key, idx, st: CondorGMMState, prev_st: CondorGMMState, hypers: Hyperparams, c: LogConfig
) -> tuple[Gaussian, dict]:
    k1, k2 = split(key, 2)
    masked_datapoints = get_relevant_datapoints_for_gaussian(idx, st, hypers)
    gaussian = st.gaussians[idx]

    xyz_params_if_new = model.get_new_gaussian_xyz_params(
        idx, st.matter.background_initialization_params, st.scene, hypers
    )
    xyz_params_if_evolved = model.get_evolve_xyz_prior_params(
        prev_st.gaussians[gaussian.origin],
        st.scene,
        prev_st.scene,
        prev_st.matter.background_evolution_params,
        hypers,
    )
    xyz_params = jax.tree.map(
        lambda x, y: jnp.where(gaussian.origin == -1, x, y),
        xyz_params_if_new,
        xyz_params_if_evolved,
    )
    xyz, cov = normal_my_inverse_wishart_update(
        k1, masked_datapoints.value.xyz, masked_datapoints.flag, xyz_params
    )

    rgb_params_if_new = model.get_new_gaussian_rgb_params(
        idx, st.matter.background_initialization_params, st.scene, hypers
    )
    rgb_params_if_evolved = model.get_evolve_rgb_prior_params(
        prev_st.gaussians[gaussian.origin],
        prev_st.matter.background_evolution_params,
        hypers,
    )
    rgb_params = jax.tree.map(
        lambda x, y: jnp.where(gaussian.origin == -1, x, y),
        rgb_params_if_new,
        rgb_params_if_evolved,
    )
    rgb, rgb_cov = jax.vmap(
        lambda k, channel: normal_my_inverse_gamma_update(
            k,
            masked_datapoints.value.rgb[:, channel],
            masked_datapoints.flag,
            rgb_params[channel],
        )
    )(split(k2, 3), jnp.arange(3))

    updated_gaussian = gaussian.replace(xyz=xyz, xyz_cov=cov, rgb=rgb, rgb_vars=rgb_cov)
    updated_gaussian = (
        jax.tree.map(  # If the updated gaussian has NaNs, keep the old values.
            lambda x, y: jnp.where(updated_gaussian.has_nan(), x, y),
            gaussian,
            updated_gaussian,
        )
    )
    return updated_gaussian, {}


def gibbs_on_origins(
    key,
    st: CondorGMMState,
    prev_st: CondorGMMState,
    hypers: Hyperparams,
    c: LogConfig = default_config,
):
    new_origins, meta_dict = jax.vmap(
        lambda key, idx: gibbs_on_one_origin(key, idx, st, prev_st, hypers, c)
    )(split(key, len(st.matter.gaussians)), jnp.arange(len(st.matter.gaussians)))
    new_st = st.replace({"matter": {"gaussians": {"origin": new_origins}}})
    return wrap(new_st, c, "gibbs_on_origins", meta_dict)


def gibbs_on_one_origin(
    key,
    gaussian_idx: int,
    st: CondorGMMState,
    prev_st: CondorGMMState,
    hypers: Hyperparams,
    c: LogConfig,
) -> tuple[IntArray, dict]:
    gaussian = st.gaussians[gaussian_idx]

    def score_origin(o):
        return model.generate_gaussian_i_at_noninitial_timestep.assess(
            model.gaussian_to_choicemap_for_ggiant(gaussian.replace(origin=o)),
            (
                gaussian.idx,
                st.scene,
                prev_st.scene,
                prev_st.matter,
                prev_st.gaussian_has_assoc_mask,
                hypers,
            ),
        )[0]

    origins_to_try = jnp.array([-1, gaussian_idx], dtype=int)
    scores = jax.vmap(score_origin)(origins_to_try)
    new_origin = origins_to_try[genjax.categorical(scores)(key)]
    return new_origin, {}


def _gibbs_on_parameter(
    key,
    domain: Domain,
    gaussians_with_assoc: Mask[Gaussian],
    gaussian_to_score: Callable[[Gaussian, FloatArray], FloatArray],
    hypers: Hyperparams,
    c: LogConfig = default_config,
) -> tuple[FloatFromDiscreteSet, dict]:
    def score_value(val: FloatFromDiscreteSet):
        scores = jax.vmap(lambda g: gaussian_to_score(g, val.value))(
            gaussians_with_assoc.value
        )
        is_bkg = jax.vmap(lambda gidx: model.gaussian_is_background(gidx, hypers))(
            gaussians_with_assoc.value.idx
        )
        has_prev = gaussians_with_assoc.value.origin != -1
        mask = jnp.logical_and(
            jnp.logical_and(jnp.asarray(gaussians_with_assoc.flag), is_bkg), has_prev
        )
        scores = jnp.where(mask, scores, jnp.zeros_like(scores))
        return scores.sum(), {
            "per_gaussian": scores
        } if c.log_global_param_move_details else {}

    scores, per_gaussian = jax.vmap(score_value)(domain.discrete_float_values)
    idx = genjax.categorical(scores)(key)
    return FloatFromDiscreteSet(idx, domain), {
        "scores": scores,
        **per_gaussian,
        "has_assoc": gaussians_with_assoc.flag,
    } if c.log_global_param_move_details else {}


def gibbs_on_evolution_params(
    key,
    st: CondorGMMState,
    prev_st: CondorGMMState,
    hypers: Hyperparams,
    c: LogConfig = default_config,
) -> tuple[CondorGMMState, Metadata]:
    if not hypers.infer_background_evolution_params:
        return wrap(st, c, "gibbs_on_evolution_params")

    k1, k2, k3, k4 = split(key, 4)
    doms = hypers.evolved_gaussian_prior_param_domains
    current_params = st.matter.background_evolution_params

    # Sample prob_gaussian_is_new
    prob_is_new, scores1 = _gibbs_on_parameter(
        k1,
        doms.prob_gaussian_is_new_domain,
        st.gaussians_with_assoc,
        lambda g, prob: model.generate_origin.assess(
            model.origin_to_choicemap(g.origin),
            (
                g.idx,
                current_params.replace(
                    prob_gaussian_is_new=doms.prob_gaussian_is_new_domain.first_value_above(
                        prob
                    )
                ),
                hypers,
            ),
        )[0],
        hypers,
        c,
    )

    # Sample xyz_cov_pcnt
    xyz_cov_pcnt, scores2 = _gibbs_on_parameter(
        k2,
        doms.xyz_cov_pcnt_domain,
        st.gaussians_with_assoc,
        lambda g, pcnt: my_inverse_wishart.logpdf(
            g.xyz_cov,
            pcnt,
            model.get_evolve_xyz_prior_params(
                prev_st.gaussians[g.origin],
                st.scene,
                prev_st.scene,
                current_params.replace(
                    xyz_cov_pcnt=doms.xyz_cov_pcnt_domain.first_value_above(pcnt)
                ),
                hypers,
            ).prior_cov,
        ),
        hypers,
        c,
    )

    # Sample rgb_var_pcnt
    rgb_var_pcnt, scores3 = _gibbs_on_parameter(
        k3,
        doms.rgb_var_pcnt_domain,
        st.gaussians_with_assoc,
        lambda g, pcnt: jnp.sum(
            jax.vmap(
                lambda var, prior_var: my_inverse_gamma.logpdf(var, pcnt, prior_var)
            )(
                g.rgb_vars,
                model.get_evolve_rgb_prior_params(
                    prev_st.gaussians[g.origin],
                    current_params.replace(
                        rgb_var_pcnt=doms.rgb_var_pcnt_domain.first_value_above(pcnt)
                    ),
                    hypers,
                ).prior_var,
            )
        ),
        hypers,
        c,
    )

    # Sample target_xyz_mean_std
    target_xyz_mean_std, scores4 = _gibbs_on_parameter(
        k4,
        doms.target_xyz_mean_std_domain,
        st.gaussians_with_assoc,
        lambda g, std: genjax.mv_normal.logpdf(
            g.xyz,
            model.get_evolve_xyz_prior_params(
                prev_st.gaussians[g.origin],
                st.scene,
                prev_st.scene,
                current_params.replace(
                    target_xyz_mean_std=doms.target_xyz_mean_std_domain.first_value_above(
                        std
                    )
                ),
                hypers,
            ).prior_mean,
            g.xyz_cov
            / (jnp.sort(cov_to_isovars_and_quaternion(g.xyz_cov)[0])[1] / (std**2)),
        ),
        hypers,
        c,
    )

    # Create new evolution params
    new_params = current_params.replace(
        prob_gaussian_is_new=prob_is_new,
        xyz_cov_pcnt=xyz_cov_pcnt,
        rgb_var_pcnt=rgb_var_pcnt,
        target_xyz_mean_std=target_xyz_mean_std,
    )

    new_st = st.replace({"matter": {"background_evolution_params": new_params}})
    return wrap(
        new_st,
        c,
        "gibbs_on_evolution_params",
        {
            "prob_is_new": scores1,
            "xyz_cov_pcnt": scores2,
            "rgb_var_pcnt": scores3,
            "target_xyz_mean_std": scores4,
        }
        if c.log_global_param_move_details
        else {},
    )

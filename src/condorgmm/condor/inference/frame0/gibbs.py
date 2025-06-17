import jax
import jax.numpy as jnp
from jax.random import split
import genjax
from genjax import Mask
from typing import Callable
from ...types import (
    Hyperparams,
    Gaussian,
    CondorGMMState,
    NewGaussianPriorParams,
    FloatFromDiscreteSet,
    Domain,
)
from ...model.distributions import my_inverse_wishart, my_inverse_gamma
from ..instrumentation import (
    LogConfig,
    Metadata,
    default_config,
    wrap,
    flatten_metadata,
    sequence,
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
from genjax.typing import FloatArray

### Top level inference functions ###


def run_n_mcmc_sweeps(
    key,
    initial_st: CondorGMMState,
    hypers: Hyperparams,
    n_mcmc_sweeps: int,
    c: LogConfig = default_config,
) -> tuple[CondorGMMState, Metadata]:
    def kernel(st: CondorGMMState, key):
        st, m = mcmc_sweep(key, st, hypers, c)
        return st, m

    final_st, batched_meta = jax.lax.scan(kernel, initial_st, split(key, n_mcmc_sweeps))
    return final_st, flatten_metadata(batched_meta)


run_n_mcmc_sweeps_jitted = jax.jit(
    run_n_mcmc_sweeps, static_argnames=("n_mcmc_sweeps",)
)


def mcmc_sweep(
    key, st: CondorGMMState, hypers: Hyperparams, c: LogConfig = default_config
) -> tuple[CondorGMMState, Metadata]:
    k1, k2, k3, k4, k5 = split(key, 5)
    st, m1 = gibbs_on_mixture_weights(k1, st, hypers, c)
    st, m2 = gibbs_on_gaussian_params(k2, st, hypers, c)
    st, m3 = gibbs_on_gaussian_prior_params(k3, st, hypers, c)
    st, m4 = update_tiling(k4, st, hypers, c)
    st, m5 = update_datapoint_associations_and_depths(k5, st, hypers, c)
    return st, sequence(m1, m2, m3, m4, m5)


### Mixture weight inference ###


def gibbs_on_mixture_weights(
    key, st: CondorGMMState, hypers: Hyperparams, c: LogConfig
) -> tuple[CondorGMMState, Metadata]:
    weights = mixture_weights_to_categorical_update(
        key,
        st.datapoints.value.gaussian_idx,
        st.datapoints.flag,
        jax.vmap(lambda i: model.new_gaussian_mixture_weight_alpha(i, True, hypers))(
            jnp.arange(len(st.gaussians))
        ),
    )
    new_st = st.replace({"matter": {"gaussians": {"mixture_weight": weights}}})
    return wrap(new_st, c, "gibbs_on_mixture_weights")


### Gaussian parameter update ###


def gibbs_on_gaussian_params(
    key, st: CondorGMMState, hypers: Hyperparams, c: LogConfig
) -> tuple[CondorGMMState, Metadata]:
    new_gaussians, meta_dict = jax.vmap(
        lambda key, idx: gibbs_on_one_gaussian(key, idx, st, hypers, c)
    )(split(key, len(st.matter.gaussians)), jnp.arange(len(st.matter.gaussians)))
    new_st = st.replace({"matter": {"gaussians": new_gaussians}})
    return wrap(new_st, c, "gibbs_on_gaussian_params", meta_dict)


def gibbs_on_one_gaussian(
    key, idx, st: CondorGMMState, hypers: Hyperparams, c: LogConfig
) -> tuple[Gaussian, dict]:
    k1, k2 = split(key, 2)
    masked_datapoints = get_relevant_datapoints_for_gaussian(idx, st, hypers)
    gaussian = st.gaussians[idx]
    params = st.matter.background_initialization_params
    xyz_params = model.get_new_gaussian_xyz_params(idx, params, st.scene, hypers)
    xyz, cov = normal_my_inverse_wishart_update(
        k1, masked_datapoints.value.xyz, masked_datapoints.flag, xyz_params
    )
    rgb_params = model.get_new_gaussian_rgb_params(idx, params, st.scene, hypers)
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


### Global parameter inference ###


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
        scores = jnp.where(  # type: ignore
            jnp.logical_and(gaussians_with_assoc.flag, is_bkg),  # type: ignore
            scores,
            0.0,  # type: ignore
        )
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


def _gibbs_on_prior_cov_in_channel(
    key,
    ch,
    prior_cov,
    xyz_cov_pcnt,
    cov_domain,
    gaussians_with_assoc: Mask[Gaussian],
    hypers,
    c: LogConfig,
) -> tuple[FloatFromDiscreteSet, jnp.ndarray, dict]:
    std, scores = _gibbs_on_parameter(
        key,
        cov_domain,
        gaussians_with_assoc,
        lambda g, std: my_inverse_wishart.logpdf(
            g.xyz_cov, xyz_cov_pcnt, prior_cov.at[ch, ch].set(std**2)
        ),
        hypers,
        c,
    )
    prior_cov = prior_cov.at[ch, ch].set(std.value**2)
    return (
        std,
        prior_cov,
        {f"xyzcovprior-ch{ch}": scores} if c.log_global_param_move_details else {},
    )


def _gibbs_on_xyz_cov_prior_params(
    key,
    st: CondorGMMState,
    hypers: Hyperparams,
    c: LogConfig,
    #            unbatched           batched (3,)
) -> tuple[FloatFromDiscreteSet, FloatFromDiscreteSet, dict]:
    k1, k2, k3, k4 = split(key, 4)
    doms = hypers.prior_param_domains
    prev = st.matter.background_initialization_params.values

    xyz_cov_pcnt, scores = _gibbs_on_parameter(
        k1,
        doms.cov_pseudocount_domain_3d,
        st.gaussians_with_assoc,
        lambda g, pcnt: my_inverse_wishart.logpdf(g.xyz_cov, pcnt, prev.xyz_prior_cov),
        hypers,
        c,
    )

    prior_cov = prev.xyz_prior_cov
    pcnt = xyz_cov_pcnt.value
    gs = st.gaussians_with_assoc
    dom = doms.xyz_pseudo_std_domain
    std1, prior_cov, d1 = _gibbs_on_prior_cov_in_channel(
        k2, 0, prior_cov, pcnt, dom, gs, hypers, c
    )
    std2, prior_cov, d2 = _gibbs_on_prior_cov_in_channel(
        k3, 1, prior_cov, pcnt, dom, gs, hypers, c
    )
    std3, prior_cov, d3 = _gibbs_on_prior_cov_in_channel(
        k4, 2, prior_cov, pcnt, dom, gs, hypers, c
    )
    xyz_iso_stds = jax.tree.map(lambda x, y, z: jnp.stack([x, y, z]), std1, std2, std3)
    return (
        xyz_cov_pcnt,
        xyz_iso_stds,
        (
            {"xyzcovpcnt": scores, **d1, **d2, **d3}
            if c.log_global_param_move_details
            else {}
        ),
    )


def _gibbs_on_rgb_prior_params(
    key,
    ch: int,
    st: CondorGMMState,
    hypers: Hyperparams,
    c: LogConfig,
    #            unbatched           batched (3,)           unbatched
) -> tuple[FloatFromDiscreteSet, FloatFromDiscreteSet, FloatFromDiscreteSet, dict]:
    k1, k2, k3 = split(key, 3)
    doms = hypers.prior_param_domains
    prev = st.matter.background_initialization_params.values
    pcnt, sc1 = _gibbs_on_parameter(
        k1,
        doms.std_pseudocount_domain_1d,
        st.gaussians_with_assoc,
        lambda g, pcnt: my_inverse_gamma.logpdf(
            g.rgb_vars[ch], pcnt, prev.rgb_var_pseudo_sample_vars[ch]
        ),
        hypers,
        c,
    )
    std, sc2 = _gibbs_on_parameter(
        k2,
        doms.rgb_pseudo_std_domain,
        st.gaussians_with_assoc,
        lambda g, std: my_inverse_gamma.logpdf(g.rgb_vars[ch], pcnt.value, std**2),
        hypers,
        c,
    )
    # This mean_pcnt update can be done in parallel to the (pcnt -> std) update.
    mean_pcnt, sc3 = _gibbs_on_parameter(
        k3,
        doms.mean_pseudocount_domain,
        st.gaussians_with_assoc,
        lambda g, meanpcnt: genjax.normal.logpdf(
            g.rgb[ch], prev.rgb_mean_center[ch], jnp.sqrt(g.rgb_vars[ch] / meanpcnt)
        ),
        hypers,
        c,
    )
    return (
        pcnt,
        std,
        mean_pcnt,
        (
            {"rgbvarpcnt": sc1, "rgbvarstd": sc2, "rgbmeanpcnt": sc3}
            if c.log_global_param_move_details
            else {}
        ),
    )


def gibbs_on_gaussian_prior_params(
    key, st: CondorGMMState, hypers: Hyperparams, c: LogConfig
) -> tuple[CondorGMMState, Metadata]:
    k1, k2, k3 = split(key, 3)

    # Notice that these 3 groups of updates can each be run in parallel.
    xyz_cov_pcnt, xyz_iso_stds, md1 = _gibbs_on_xyz_cov_prior_params(k1, st, hypers, c)
    xyz_mean = st.scene.transform_World_Camera.inv().pos
    xyz_mean_pcnt, sc = _gibbs_on_parameter(
        k2,
        hypers.prior_param_domains.mean_pseudocount_domain,
        st.gaussians_with_assoc,
        lambda g, pcnt: genjax.mv_normal.logpdf(g.xyz, xyz_mean, g.xyz_cov / pcnt),
        hypers,
        c,
    )
    rgb_var_pcnt, rgb_prior_std, rgb_mean_pcnt, md2 = jax.vmap(
        lambda k, channel: _gibbs_on_rgb_prior_params(k, channel, st, hypers, c)
    )(split(k3, 3), jnp.arange(3))

    new_params = NewGaussianPriorParams(
        xyz_cov_pcnt=xyz_cov_pcnt,
        xyz_cov_isotropic_prior_stds=xyz_iso_stds,
        xyz_mean_pcnt=xyz_mean_pcnt,
        rgb_var_n_pseudo_obs=rgb_var_pcnt,
        rgb_var_pseudo_sample_stds=rgb_prior_std,
        rgb_mean_n_pseudo_obs=rgb_mean_pcnt,
    )
    new_st = st.replace({"matter": {"background_initialization_params": new_params}})
    return wrap(
        new_st,
        c,
        "gibbs_on_gaussian_prior_params",
        {
            **md1,
            **{"xyzmeanpcnt": sc},
            **md2,
        }
        if c.log_global_param_move_details
        else {},
    )

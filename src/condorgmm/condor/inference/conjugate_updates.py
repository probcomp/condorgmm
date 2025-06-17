import jax
import jax.numpy as jnp
from genjax import mv_normal, normal
from ..model.distributions import dirichlet, gamma, my_inverse_wishart, my_inverse_gamma
from ..types import GAMMA_RATE_PARAMETER, NIWParams, NIGParams
from genjax.typing import BoolArray, Array

## Dirichlet / Categorical ##


def _get_counts(values: Array, mask: BoolArray, N):
    return jnp.bincount(jnp.where(mask, values + 1, -1), length=N + 1)[1:]


def mixture_weights_to_categorical_update_from_counts(
    key,
    counts,  # (N,) int array of counts
    dirichlet_alphas,  # (N,) float array of prior alphas for dirichlet
):
    k1, k2 = jax.random.split(key)
    pvec = dirichlet(dirichlet_alphas + counts)(k1)
    total = gamma(dirichlet_alphas.sum(), GAMMA_RATE_PARAMETER)(k2)
    return total * pvec


def mixture_weights_to_categorical_update(
    key,
    index_samples,  # (K,) int array of indices
    index_samples_mask,  # (K,) boolean mask
    dirichlet_alphas,  # (N,) float array of prior alphas for dirichlet
):
    return mixture_weights_to_categorical_update_from_counts(
        key,
        _get_counts(index_samples, index_samples_mask, len(dirichlet_alphas)),
        dirichlet_alphas,
    )


## Normal / Normal-Inverse-Wishart ##


def normal_my_inverse_wishart_update_params(
    obs: Array,  # (N, D) array of observations
    obs_mask: BoolArray,  # (N,) boolean mask
    params: NIWParams,
):
    (N, D) = obs.shape

    n_obs = jnp.sum(obs_mask)
    sum_of_obs = jnp.sum(jnp.where(obs_mask[:, None], obs, 0), axis=0)
    mean_obs = sum_of_obs / n_obs

    prior_Psi = params.cov_pcnt * params.prior_cov
    assert prior_Psi.shape == (D, D)

    updated_mean = (params.mean_pcnt * params.prior_mean + sum_of_obs) / (
        params.mean_pcnt + n_obs
    )
    assert params.prior_mean.shape == (D,)
    assert sum_of_obs.shape == (D,)
    assert updated_mean.shape == (D,)

    empirical_err = jnp.where(obs_mask[:, None], obs - mean_obs, 0)  # (N, D)
    assert empirical_err.shape == (N, D)
    empirical_cov = empirical_err.T @ empirical_err  # (D, D)
    assert empirical_cov.shape == (D, D)
    cov_from_means = (
        (params.mean_pcnt * n_obs)
        / (params.mean_pcnt + n_obs)
        * (
            (mean_obs - params.prior_mean)[:, None]
            @ (mean_obs - params.prior_mean)[None, :]  # (D, D)
        )
    )
    assert cov_from_means.shape == (D, D)
    updated_cov = (prior_Psi + empirical_cov + cov_from_means) / (
        params.cov_pcnt + n_obs
    )
    assert updated_cov.shape == (D, D)

    return (
        params.mean_pcnt + n_obs,
        jnp.where(n_obs == 0, params.prior_mean, updated_mean),
        params.cov_pcnt + n_obs,
        jnp.where(n_obs == 0, params.prior_cov, updated_cov),
    )


def normal_my_inverse_wishart_update(
    key,
    obs,  # (N, D) array of observations
    obs_mask,  # (N,) boolean mask
    params: NIWParams,
):
    k1, k2 = jax.random.split(key)
    (new_mean_pcnt, updated_mean, new_cov_pcnt, updated_cov) = (
        normal_my_inverse_wishart_update_params(obs, obs_mask, params)
    )
    cov = my_inverse_wishart(new_cov_pcnt, updated_cov)(k1)
    mean = mv_normal(updated_mean, cov / new_mean_pcnt)(k2)
    return mean, cov


## Normal / Normal-Inverse-Gamma ##


def normal_my_inverse_gamma_update(
    key,
    obs,  # (N,) array of observations
    obs_mask,  # (N,) boolean mask
    params: NIGParams,
):
    k1, k2 = jax.random.split(key)
    (new_mean_pcnt, updated_mean, new_cov_pcnt, updated_cov) = (
        normal_my_inverse_wishart_update_params(
            obs[:, None],
            obs_mask,
            NIWParams(
                cov_pcnt=params.var_pcnt,
                prior_cov=jnp.array([[params.prior_var]]),
                mean_pcnt=params.mean_pcnt,
                prior_mean=jnp.array([params.prior_mean]),
            ),
        )
    )
    var = my_inverse_gamma(new_cov_pcnt, updated_cov[0, 0])(k1)
    mean = normal(updated_mean[0], jnp.sqrt(var / new_mean_pcnt))(k2)
    return mean, var

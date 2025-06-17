import jax
from jax.random import split
from jax.random import key as prngkey
import jax.numpy as jnp
from genjax import mv_normal
from condorgmm.condor.types import NIWParams
from condorgmm.condor.inference.conjugate_updates import (
    normal_my_inverse_wishart_update_params,
    normal_my_inverse_wishart_update,
)
from condorgmm.condor.model.distributions import my_inverse_wishart


def is_close(x, y):
    return jnp.logical_or(jnp.allclose(x, y, rtol=3e-1), jnp.allclose(x, y, atol=1e-1))


def all_close(x, y):
    return jnp.all(jax.vmap(is_close)(x.flatten(), y.flatten()))


def _generate_niw_params(key):
    k1, k2, k3, k4 = split(key, 4)
    cov_pcnt = 3 + jnp.abs(jax.random.normal(k1))
    cov_prior_pobs = jax.random.normal(k2, (3, 3))
    cov_prior_pobs = cov_prior_pobs @ cov_prior_pobs.T
    mean_pcnt = jnp.abs(2 * jax.random.normal(k3))
    mean_prior_pobs = jax.random.normal(k4, (3,))
    return NIWParams(
        cov_pcnt=cov_pcnt,
        prior_cov=cov_prior_pobs,
        mean_pcnt=mean_pcnt,
        prior_mean=mean_prior_pobs,
    )


def _generate_testcase(n_obs, key, cnt=0):
    k1, k2, k3, k4, k5 = split(key, 5)
    params = _generate_niw_params(k1)
    cov = my_inverse_wishart(params.cov_pcnt, params.prior_cov)(k2)
    mean = mv_normal(params.prior_mean, cov / params.mean_pcnt)(k3)
    garbage_values = jax.random.normal(k4, (n_obs, 3))
    garbage_indices = jax.random.randint(k5, (n_obs,), 0, n_obs * 2)
    return (cov, mean, params, n_obs, garbage_values, garbage_indices)


def _generate_masked_obs(cov, mean, n_obs, garbage_values, idxs_for_garbage_values):
    obs = jax.vmap(mv_normal(mean, cov))(split(prngkey(0), n_obs + len(garbage_values)))
    obs = obs.at[idxs_for_garbage_values].set(garbage_values)
    mask = (
        jnp.ones(n_obs + len(garbage_values), dtype=bool)
        .at[idxs_for_garbage_values]
        .set(jnp.array(0, dtype=bool))
    )
    return obs, mask


def check_normal_inverse_wishart_parameter_recovery(
    cov,  # (3, 3)
    mean,  # (3,)
    prior_params: NIWParams,
    n_obs,  # int > 100
    garbage_values,  # (n_garbage, 3)
    idxs_for_garbage_values,  # (n_garbage,) int array
    recovery_method,  # function
):
    """
    Tests that (cov, mean) are recovered by the normal inverse wishart conjugate update.

    Args:
        - cov, mean: parameters to recover
        - cov_pcnt, mean_pcnt, cov_prior_pobs, mean_prior_pobs: NIW prior parameters
        - n_obs: number of observations from which to try to recover the parameters
        - random_values: garbage values to add, masked out, to the observations
            (to test that masked observations are handled appropriately)
        - idxs_for_random_values: indices of the garbage values in the overall
            masked (n_obs + n_garbage,) array of observations
    """
    obs, mask = _generate_masked_obs(
        cov, mean, n_obs, garbage_values, idxs_for_garbage_values
    )
    recovered_mean, recovered_cov = recovery_method(obs, mask, prior_params)

    failstr = f"Failed with params={prior_params}, mean={mean}, cov={cov}"
    assert all_close(mean, recovered_mean), failstr
    if not all_close(cov, recovered_cov):
        print(recovered_cov / cov)
        assert False, failstr


def recover_via_params(obs, mask, prior_params):
    _, recovered_mean, _, recovered_cov = normal_my_inverse_wishart_update_params(
        obs, mask, prior_params
    )
    return recovered_mean, recovered_cov


def recover_via_update(key):
    def recover(obs, mask, prior_params):
        return normal_my_inverse_wishart_update(key, obs, mask, prior_params)

    return recover


def test_normal_inverse_wishart_parameter_recovery():
    n_obs = 10_000
    key = prngkey(0)
    for _ in range(10):
        key, k2 = split(key)
        testcase = _generate_testcase(n_obs, key)
        check_normal_inverse_wishart_parameter_recovery(
            *testcase, recovery_method=recover_via_params
        )
        check_normal_inverse_wishart_parameter_recovery(
            *testcase, recovery_method=recover_via_update(k2)
        )


def test_normal_inverse_wishart_params_on_no_obs():
    key = prngkey(0)
    for _ in range(10):
        key, k2 = split(key)
        (cov, mean, params, n_obs, garbage_values, garbage_indices) = (
            _generate_testcase(0, key)
        )
        obs, mask = _generate_masked_obs(
            cov, mean, n_obs, garbage_values, garbage_indices
        )
        recovered_mean, recovered_cov = recover_via_params(obs, mask, params)
        assert jnp.all(recovered_mean == params.prior_mean)
        assert jnp.all(recovered_cov == params.prior_cov)

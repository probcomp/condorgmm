import pytest
import jax.numpy as jnp
from condorgmm.condor.model.distributions.variance_priors import (
    inverse_wishart,
    wishart,
    my_inverse_gamma,
    my_inverse_wishart,
)
import genjax
from condorgmm.condor.model.distributions import gamma


def check_gamma_wishart_logpdf_equivalence(x, alpha, beta):
    nu = 2 * alpha
    Lambda = jnp.array([[1 / (2 * beta)]])
    assert jnp.allclose(
        gamma.logpdf(x, alpha, beta),
        wishart.logpdf(jnp.array([[x]]), nu, Lambda),
    )


def check_inversegamma_inversewishart_logpdf_equivalence(x, alpha, beta):
    nu = 2 * alpha
    Lambda = jnp.array([[1 / (2 * beta)]])
    Psi = jnp.linalg.inv(Lambda)
    X = jnp.array([[x]])
    assert jnp.allclose(
        genjax.inverse_gamma.logpdf(x, alpha, beta),
        inverse_wishart.logpdf(X, nu, Psi),
    )


def check_my_inversegamma_inversewishart_logpdf_equivalence(
    x, n_pseudo_obs, pseudo_var
):
    # Convert inverse gamma parameters to inverse wishart parameters
    pseudo_cov = jnp.eye(1) * pseudo_var

    assert jnp.allclose(
        my_inverse_gamma.logpdf(x, n_pseudo_obs, pseudo_var),
        my_inverse_wishart.logpdf(jnp.array([[x]]), n_pseudo_obs, pseudo_cov),
    )


@pytest.mark.parametrize(
    "x, alpha, beta",
    [
        (0.5, 2.0, 2.0),
        (1.0, 3.0, 1.0),
        (2.0, 4.0, 0.5),
        (0.1, 1.5, 1.5),
        (1.5, 2.5, 2.5),
    ],
)
def test_gamma_wishart_logpdf_equivalence(x, alpha, beta):
    check_gamma_wishart_logpdf_equivalence(x, alpha, beta)


@pytest.mark.parametrize(
    "x, alpha, beta",
    [
        (0.5, 2.0, 2.0),
        (1.0, 3.0, 1.0),
        (2.0, 4.0, 0.5),
        (0.1, 1.5, 1.5),
        (1.5, 2.5, 2.5),
    ],
)
def test_inversegamma_inversewishart_logpdf_equivalence(x, alpha, beta):
    check_inversegamma_inversewishart_logpdf_equivalence(x, alpha, beta)


@pytest.mark.parametrize(
    "x, n_pseudo_obs, pseudo_var",
    [
        (0.5, 4.0, 1.0),  # standard case
        (0.1, 3.0, 0.5),  # small x
        (2.0, 8.0, 2.0),  # large n_pseudo_obs
        (1.5, 5.0, 0.1),  # small pseudo_var
        (1.0, 6.0, 5.0),  # large pseudo_var
    ],
)
def test_my_inversegamma_inversewishart_logpdf_equivalence(x, n_pseudo_obs, pseudo_var):
    check_my_inversegamma_inversewishart_logpdf_equivalence(x, n_pseudo_obs, pseudo_var)

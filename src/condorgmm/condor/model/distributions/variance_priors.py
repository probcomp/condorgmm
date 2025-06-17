import jax
import jax.numpy as jnp
import genjax
from genjax import Pytree
import tensorflow_probability.substrates.jax.distributions as tfd
from ...utils import MyPytree
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, FloatLogSlider


def truncate_eigenval_ratio(matrix, threshold=1e3):
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    max_eval = eigvals[-1]
    eigvals_over_max = eigvals / max_eval
    eigvals_over_max = jnp.where(
        eigvals_over_max < 1 / threshold, 1 / threshold, eigvals_over_max
    )
    eigvals = eigvals_over_max * max_eval
    new_matrix = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
    return new_matrix


@Pytree.dataclass
class MyInverseGamma(MyPytree, genjax.ExactDensity):
    @staticmethod
    def _to_alpha_beta(n_pseudo_observations, pseudo_sample_variance):
        alpha = n_pseudo_observations / 2
        beta = pseudo_sample_variance * alpha
        return alpha, beta

    def sample(self, key, n_pseudo_observations, pseudo_sample_variance):
        alpha, beta = self._to_alpha_beta(n_pseudo_observations, pseudo_sample_variance)
        return genjax.inverse_gamma.sample(key, alpha, beta)

    def logpdf(self, x, n_pseudo_observations, pseudo_sample_variance):
        alpha, beta = self._to_alpha_beta(n_pseudo_observations, pseudo_sample_variance)
        return genjax.inverse_gamma.logpdf(x, alpha, beta)


my_inverse_gamma = MyInverseGamma()


@Pytree.dataclass
class Wishart(MyPytree, genjax.ExactDensity):
    @staticmethod
    def _tfp_distr(nu, Lambda):
        scale_tril = jnp.linalg.cholesky(Lambda)
        return tfd.WishartTriL(nu, scale_tril)

    def sample(self, key, nu, Lambda):
        return self._tfp_distr(nu, Lambda).sample(seed=key)

    def logpdf(self, x, nu, Lambda):
        return self._tfp_distr(nu, Lambda).log_prob(x)


wishart = Wishart()


@Pytree.dataclass
class InverseWishart(MyPytree, genjax.ExactDensity):
    def sample(self, key, nu, Psi):
        W = wishart.sample(key, nu, jnp.linalg.inv(Psi))
        mtx = jnp.linalg.inv(W)
        return truncate_eigenval_ratio(mtx)

    def logpdf(self, X, nu, Psi):
        # TODO: I have not yet tested this Jacobian correction except in the 1D case.
        d = X.shape[0]
        W = jnp.linalg.inv(X)
        wishart_logpdf = wishart.logpdf(W, nu, jnp.linalg.inv(Psi))
        return wishart_logpdf - (d + 1) * jnp.linalg.slogdet(X)[1]


inverse_wishart = InverseWishart()


@Pytree.dataclass
class MyInverseWishart(MyPytree, genjax.ExactDensity):
    @staticmethod
    def _get_nu_Psi(n_pseudo_observations, pseudo_sample_cov):
        nu = n_pseudo_observations
        Psi = pseudo_sample_cov * nu
        return nu, Psi

    def sample(self, key, n_pseudo_observations, pseudo_sample_cov):
        nu, Psi = self._get_nu_Psi(n_pseudo_observations, pseudo_sample_cov)
        return inverse_wishart.sample(key, nu, Psi)

    def logpdf(self, X, n_pseudo_observations, pseudo_sample_cov):
        nu, Psi = self._get_nu_Psi(n_pseudo_observations, pseudo_sample_cov)
        return inverse_wishart.logpdf(X, nu, Psi)


my_inverse_wishart = MyInverseWishart()

### IPYWidgets for playing with some of these distributions ###


def get_my_inverse_gamma_widget(
    nxs=1000,
    minx=1e-5,
    maxx=1e5,
    min_npseudo=1e-5,
    max_npseudo=1e5,
    min_psv=1e-5,
    max_psv=1e5,
    plot_sqrt_x=False,
):
    xs = jnp.logspace(jnp.log10(minx), jnp.log10(maxx), nxs)

    def plot_my_inverse_gamma(n_pseudo_obs, pseudo_sample_variance):
        pdf = jax.vmap(
            lambda x: jnp.exp(
                my_inverse_gamma.logpdf(x, n_pseudo_obs, pseudo_sample_variance)
            )
        )(xs)
        # Adjust the density for the log scale
        adjusted_pdf = pdf * xs * 2 if plot_sqrt_x else pdf * xs
        # TODO: double check this correction is right for plot_sqrt_x = True

        plt.figure(figsize=(8, 5))
        plt.plot(jnp.where(plot_sqrt_x, jnp.sqrt(xs), xs), adjusted_pdf)
        plt.axvline(
            jnp.where(  # type: ignore
                plot_sqrt_x, jnp.sqrt(pseudo_sample_variance), pseudo_sample_variance
            ),
            color="red",
            linestyle="--",
            label="pseudo_sample_variance"
            if not plot_sqrt_x
            else "sqrt(pseudo_sample_variance)",
        )
        plt.xscale("log")
        plt.yscale("linear")
        plt.xlabel("x" if not plot_sqrt_x else "sqrt(x)")
        plt.ylabel("Density (adjusted for log scale)")
        plt.title("MyInverseGamma Distribution")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.show()

    return interact(
        plot_my_inverse_gamma,
        n_pseudo_obs=widgets.FloatLogSlider(
            min=jnp.log10(min_npseudo), max=jnp.log10(max_npseudo), step=0.1, value=1
        ),
        pseudo_sample_variance=FloatLogSlider(
            min=jnp.log10(min_psv),
            max=jnp.log10(max_psv),
            step=0.1,
            value=1,
            description="pseudo_sample_variance"
            if not plot_sqrt_x
            else "sqrt(pseudo_sample_variance)",
        ),
    )

import jax
import jax.numpy as jnp
import genjax
from tensorflow_probability.substrates import jax as tfp

from ...types import Domain, FloatFromDiscreteSet, MyPytree
from .discrete_laplace import discretized_laplace, index_space_discretized_laplace  # noqa: F401
from .variance_priors import inverse_wishart, my_inverse_gamma, my_inverse_wishart  # noqa: F401

### Other distributions ###

_gamma = genjax.tfp_distribution(tfp.distributions.Gamma)


def sample_gamma_safe(key, alpha, beta):
    sample = _gamma.sample(key, alpha, beta)
    return jnp.where(sample == 0, 1e-12, sample)


gamma = genjax.exact_density(sample_gamma_safe, _gamma.logpdf, "gamma")


def sample_dirichlet_safe(key, alpha):
    if alpha.shape == (1,):
        return jnp.array([1.0], dtype=jnp.float32)
    sample = genjax.dirichlet.sample(key, alpha)
    return jnp.where(sample == 0, 1e-12, sample)


def logpdf_dirichlet_safe(val, alpha):
    if alpha.shape == (1,):
        return jnp.array([0.0], dtype=jnp.float32)
    return genjax.dirichlet.logpdf(val, alpha)


dirichlet = genjax.exact_density(
    sample_dirichlet_safe, logpdf_dirichlet_safe, "dirichlet"
)


@genjax.Pytree.dataclass
class UniformFromDomain(MyPytree, genjax.ExactDensity):
    def sample(self, key, domain: Domain) -> FloatFromDiscreteSet:
        idx = jax.random.randint(key, (), 0, len(domain))
        return FloatFromDiscreteSet(idx=idx, domain=domain)

    def logpdf(self, val: FloatFromDiscreteSet, domain: Domain):
        assert val.domain == domain
        return -jnp.log(len(domain))


uniform_from_domain = UniformFromDomain()

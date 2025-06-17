import jax
import jax.numpy as jnp
from genjax import Const
import condorgmm.condor.model.distributions.discrete_laplace as dl
from condorgmm.condor.model.distributions import uniform_from_domain
from condorgmm.condor.types import Domain, FloatFromDiscreteSet
from condorgmm.condor.utils import unwrap


def test_sum_exponential_over_range():
    for minx in [0, -4, 4]:
        for maxx in [0, 5, 6]:
            for log_base in [0.5, 1.0, 2.0]:
                s1 = dl.sum_exponential_over_range(minx, maxx, log_base)
                s2 = jnp.sum(
                    jnp.array(
                        [jnp.exp(log_base * val) for val in jnp.arange(minx, maxx + 1)]
                    )
                )
                assert jnp.allclose(
                    s1, s2
                ), f"minx={minx}, maxx={maxx}, log_base={log_base}: {s1} != {s2}"


def test_normalizing_const_for_discretized_laplace():
    for minx in [0, -4, 4]:
        for maxx in [0, 5, 6]:
            for center in [jnp.array(0), jnp.array(1), jnp.array(2)]:
                for scale in [0.5, 1.0, 2.0]:
                    Z = dl.normalizing_const_for_discretized_laplace(
                        minx, maxx, center, scale
                    )
                    true_sum = jnp.sum(
                        jnp.exp(-jnp.abs(jnp.arange(minx, maxx + 1) - center) / scale)
                    )
                    assert jnp.allclose(
                        Z, true_sum
                    ), f"minx={minx}, maxx={maxx}, center={center}, scale={scale}: {Z} != {true_sum}"


def test_discretized_laplace_pdf():
    for center in [jnp.array(0), jnp.array(1), jnp.array(2)]:
        for scale in [0.5, 1.0, 2.0]:
            for minx in [Const(0), -4, Const(4)]:
                for maxx in [0, Const(5), 6]:
                    if unwrap(maxx) > unwrap(minx):
                        sum_of_pdfs = jnp.sum(
                            jnp.exp(
                                dl.discretized_laplace.logpdf(
                                    jnp.arange(unwrap(minx), unwrap(maxx) + 1),
                                    center,
                                    scale,
                                    minx,
                                    maxx,
                                )
                            )
                        )
                        assert jnp.allclose(
                            sum_of_pdfs, 1.0
                        ), f"center={center}, scale={scale}, minx={minx}, maxx={maxx}: {sum_of_pdfs} != 1.0"
                    for x in [0, -4, 4]:
                        pdf = jnp.exp(
                            dl.discretized_laplace.logpdf(x, center, scale, minx, maxx)
                        )
                        true_pdf = jnp.exp(
                            -jnp.abs(x - center) / scale
                        ) / dl.normalizing_const_for_discretized_laplace(
                            unwrap(minx), unwrap(maxx), center, scale
                        )
                        assert jnp.allclose(
                            pdf, true_pdf
                        ), f"x={x}, center={center}, scale={scale}, minx={minx}, maxx={maxx}: {pdf} != {true_pdf}"


def test_discretized_laplace_sampling():
    samples = jax.vmap(
        lambda key: dl.discretized_laplace.sample(key, 0, 4, Const(-10), Const(10))
    )(jax.random.split(jax.random.PRNGKey(0), 10000))
    empirical_pdf = jnp.bincount(samples + 10, minlength=21) / 10000
    true_pdf = jnp.exp(
        dl.discretized_laplace.logpdf(jnp.arange(-10, 11), 0, 4, -10, 10)
    )
    assert jnp.allclose(
        empirical_pdf, true_pdf, atol=0.02
    ), f"{empirical_pdf} != {true_pdf}"


def test_index_space_discretized_laplace():
    domain = Domain(jnp.logspace(-3, 1, 32, base=10))
    mean = domain.first_value_above(0.1)
    scale = 4
    logpdfs = jax.vmap(
        lambda x: dl.index_space_discretized_laplace.logpdf(x, mean, scale)
    )(FloatFromDiscreteSet(jnp.arange(32), domain))
    dl_logpdfs = jax.vmap(
        lambda idx: dl.discretized_laplace.logpdf(
            idx, mean.idx, scale, Const(0), Const(31)
        )
    )(jnp.arange(32))
    assert jnp.all(logpdfs == dl_logpdfs)
    samples = jax.vmap(
        lambda key: dl.index_space_discretized_laplace.sample(key, mean, scale)
    )(jax.random.split(jax.random.PRNGKey(0), 100))
    dl_samples = jax.vmap(
        lambda key: dl.discretized_laplace.sample(
            key, mean.idx, scale, Const(0), Const(31)
        )
    )(jax.random.split(jax.random.PRNGKey(0), 100))
    assert jnp.all(samples.idx == dl_samples)


def test_index_space_uniform():
    domain = Domain(jnp.logspace(-3, 1, 32, base=10))
    logpdfs = jax.vmap(lambda x: uniform_from_domain.logpdf(x, domain))(
        FloatFromDiscreteSet(jnp.arange(32), domain)
    )
    assert jnp.all(logpdfs == -jnp.log(32))
    samples = jax.vmap(lambda key: uniform_from_domain.sample(key, domain))(
        jax.random.split(jax.random.PRNGKey(0), 500)
    )
    assert jnp.all(samples.idx >= 0)
    assert jnp.all(samples.idx < 32)
    assert jnp.all(jnp.bincount(samples.idx, minlength=32) > 0)

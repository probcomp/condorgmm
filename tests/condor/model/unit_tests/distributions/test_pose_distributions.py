import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

import condorgmm.condor.pose as pose


def test_rotation_dists():
    assert jnp.allclose(
        pose.uniform_rot.logpdf(jnp.array([0.5, 0.5, 0.5, 0.5])), -jnp.log(jnp.pi**2)
    )

    # Check that a VMF distribution with conc \approx 0 is
    # approximately uniform and is using the same base measure
    # as the uniform pose distribution.
    assert jnp.allclose(
        pose.vmf_on_rot.logpdf(
            jnp.array([0.5, 0.5, 0.5, 0.5]), jnp.array([1.0, 0.0, 0.0, 0.0]), 1e-3
        ),
        pose.uniform_rot.logpdf(jnp.array([0.5, 0.5, 0.5, 0.5])),
        atol=1e-3,
    )

    # Check that our implementation of logpdf is consistent
    # with TFP's implementation.
    assert jnp.allclose(
        pose.vmf_on_rot.logpdf(
            jnp.array([0.5, 0.5, 0.5, 0.5]),
            jnp.array([1.0, 0.0, 0.0, 0.0]),
            1000.0,
        ),
        tfp.distributions.VonMisesFisher(
            mean_direction=jnp.array([1.0, 0.0, 0.0, 0.0]),
            concentration=1000,
        ).log_prob(jnp.array([0.5, 0.5, 0.5, 0.5])),
    )

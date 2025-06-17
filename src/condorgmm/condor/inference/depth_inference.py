import jax
import jax.numpy as jnp
import genjax


def get_tbar_proposal_parameters(
    v,  # (3,)
    mu,  # (3,)
    Sigma,  # (3, 3)
):
    f = v[2]
    vbar = v / jnp.linalg.norm(v)
    A = jnp.array(
        [
            [f / mu[2], 0, -f * mu[0] / (mu[2] ** 2)],
            [0, f / mu[2], -f * mu[1] / (mu[2] ** 2)],
        ]
    )
    b = jnp.array([f * mu[0] / mu[2], f * mu[1] / mu[2]])[:, None]

    Abar = jnp.array([A[0], A[1], vbar])
    bbar = jnp.array([b[0, 0], b[1, 0], 0])[:, None]

    v = v[:, None]  # (3, 1)
    mu = mu[:, None]  # (3, 1)

    transformed_mu = Abar @ mu + bbar
    transformed_Sigma = Abar @ Sigma @ Abar.T

    Sigma_11 = transformed_Sigma[:2, :2]  # (2, 2)
    Sigma_12 = transformed_Sigma[:2, 2:]  # (2, 1)
    Sigma_21 = transformed_Sigma[2:, :2]  # (1, 2)
    Sigma_22 = transformed_Sigma[2:, 2:]  # (1, 1)

    inverse = jnp.linalg.inv(Sigma_11)
    conditional_mean = transformed_mu[3] + Sigma_21 @ inverse @ (
        v[:2] - transformed_mu[:2]
    )
    conditional_var = Sigma_22 - Sigma_21 @ inverse @ Sigma_12
    return conditional_mean[0, 0], conditional_var[0, 0]


def get_t_proposal_parameters(v, mu, Sigma):
    tbar_mu, tbar_var = get_tbar_proposal_parameters(v, mu, Sigma)
    return tbar_mu / jnp.linalg.norm(v), tbar_var / (jnp.linalg.norm(v) ** 2)


def propose_xyz_and_estimate_p_of_image_coords(key, v, mu, Sigma):
    f = v[2]
    t_mu, t_var = get_t_proposal_parameters(v, mu, Sigma)
    proposed_t = genjax.normal(t_mu, jnp.sqrt(t_var))(key)
    proposal_density = genjax.normal.logpdf(proposed_t, t_mu, jnp.sqrt(t_var))
    proposed_xyz = proposed_t * v
    target_density = genjax.mv_normal.logpdf(proposed_xyz, mu, Sigma)
    weight = target_density - proposal_density + jnp.log(f * proposed_t**2)
    return proposed_xyz, weight


def oneparticle_is_estimate_p_of_image_coords(key, v, mu, Sigma):
    proposed_xyz, weight = propose_xyz_and_estimate_p_of_image_coords(key, v, mu, Sigma)
    return weight


def is_estimate_p_of_image_coords(key, v, mu, Sigma, n_particles):
    values = jax.vmap(
        lambda key: oneparticle_is_estimate_p_of_image_coords(key, v, mu, Sigma)
    )(jax.random.split(key, n_particles))
    return jax.scipy.special.logsumexp(values) - jnp.log(n_particles)


def linearized_estimate_p_of_image_coords(v, mu, Sigma):
    f = v[2]
    A = jnp.array(
        [
            [f / mu[2], 0, -f * mu[0] / (mu[2] ** 2)],
            [0, f / mu[2], -f * mu[1] / (mu[2] ** 2)],
        ]
    )
    b = jnp.array([f * mu[0] / mu[2], f * mu[1] / mu[2]])[:, None]
    mu2d = b + A @ mu[:, None]
    Sigma2d = A @ Sigma @ A.T
    return genjax.mv_normal.logpdf(v[:2], mu2d[:, 0], Sigma2d)


def pixel_coords_to_v(y_idx, x_idx, f, cx, cy):
    y_I = y_idx - cy + 0.5
    x_I = x_idx - cx + 0.5
    v = jnp.array([x_I, y_I, f])
    return v

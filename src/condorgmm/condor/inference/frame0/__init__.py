import jax
from jax.random import split
import jax.numpy as jnp
from genjax import Mask
from ...types import Hyperparams, Observation, CondorGMMState
from ..instrumentation import (
    LogConfig,
    sequence,
    simplemeta,
    default_config,
    prepend_label,
    Metadata,
)
from .initialization import initialize_state_with_obs_and_tiling_and_stratified_assocs
from .gibbs import (
    run_n_mcmc_sweeps,
    update_datapoint_associations_and_depths,
)
from ..shared import reinitialize_unobserved_gaussians
from typing import Tuple

update_datapoint_associations_and_depths_jitted = jax.jit(
    update_datapoint_associations_and_depths
)
run_n_mcmc_sweeps_jitted = jax.jit(
    run_n_mcmc_sweeps, static_argnames=("n_mcmc_sweeps",)
)


def run_inference(
    key,
    observations: Observation,
    hypers: Hyperparams,
    *,
    n_sweeps_per_phase: Tuple[int, ...],
    do_reinitialize_per_phase: Tuple[bool, ...],
    phase_to_add_depth_non_returns: int,
    c: LogConfig = default_config,
    given_datapoint_assignment: jnp.ndarray | None = None,
    st0=None,
) -> tuple[CondorGMMState, Metadata]:
    og_hypers = hypers

    if st0 is None:
        # Initialize the tiling, and get a CondorGMMState with
        # a stratified datapoint<>Gaussian association and the
        # observations filled in.  The latents will be placeholders values for now.
        st0 = initialize_state_with_obs_and_tiling_and_stratified_assocs(
            split(key)[0], observations, og_hypers, given_datapoint_assignment
        )
    meta = simplemeta(st0, default_config, "initial_state", {})

    st, hypers = _mask_depth_nonreturns(st0, og_hypers)
    for i in range(len(n_sweeps_per_phase)):
        key = split(key)[1]
        k1, k2, k3 = split(key, 3)

        needs_dp_update = False
        if hypers.repopulate_depth_nonreturns and i == phase_to_add_depth_non_returns:
            st = st.replace(datapoints=st0.datapoints)  # un-mask the depth nonreturns
            hypers = og_hypers
            needs_dp_update = True
            meta = sequence(meta, simplemeta(st, c, "unmask_depth_nonreturns", {}))

        if do_reinitialize_per_phase[i]:
            st = reinitialize_unobserved_gaussians(k1, st, hypers)
            m = simplemeta(st, c, "reinitialize_unobserved_gaussians", {})
            needs_dp_update = True
            meta = sequence(
                meta, prepend_label(m, f"phase_{i}:reinitialize_unobserved_gaussians:")
            )

        if needs_dp_update:
            st, m = update_datapoint_associations_and_depths_jitted(k2, st, hypers, c)
            meta = sequence(meta, prepend_label(m, f"phase_{i}:dp_update:"))

        st, m = run_n_mcmc_sweeps_jitted(k3, st, hypers, n_sweeps_per_phase[i], c)
        meta = sequence(meta, prepend_label(m, f"phase_{i}:main:"))

    return st, meta


def _mask_depth_nonreturns(
    st: CondorGMMState, hypers: Hyperparams
) -> tuple[CondorGMMState, Hyperparams]:
    mask = jnp.logical_and(hypers.datapoint_mask, st.datapoints.value.obs.depth != 0.0)
    new_hypers = hypers.replace(datapoint_mask=mask)
    new_st = st.replace(
        datapoints=Mask(
            st.datapoints.value,
            mask,
        )
    )
    return new_st, new_hypers

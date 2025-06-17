import jax
import jax.numpy as jnp
from jax.random import split
import genjax
from genjax import Mask
from ..types import Hyperparams, Gaussian, Datapoint, CondorGMMState, GAMMA_RATE_PARAMETER
from .instrumentation import (
    LogConfig,
    Metadata,
    wrap,
)
from .depth_inference import propose_xyz_and_estimate_p_of_image_coords
import condorgmm.condor.model.model as model
from genjax.typing import FloatArray


### Datapoint association + depths move ###


def update_datapoint_associations_and_depths(
    key, st: CondorGMMState, hypers: Hyperparams, c: LogConfig
) -> tuple[CondorGMMState, Metadata]:
    datapoints, meta_dict = jax.vmap(
        lambda k, idx: update_one_datapoint_association_and_depth(k, idx, st, hypers, c)
    )(split(key, len(st.datapoints.value)), jnp.arange(len(st.datapoints.value)))
    new_st = st.replace(datapoints=Mask(datapoints, st.datapoints.flag))
    return wrap(new_st, c, "update_datapoint_associations_and_depths", meta_dict)


def update_one_datapoint_association_and_depth(
    key, datapoint_idx: int, st: CondorGMMState, hypers: Hyperparams, c: LogConfig
) -> tuple[Datapoint, dict]:
    # 1. Proposal:
    #   a. For each Gaussian:
    #     i. Propose a depth value assuming this is the Gaussian, and estimate
    #       P(observed datapoint | Gaussian).
    #   b. Resample a (Gaussian, depth) pair.
    # 2. Depending on the input flags in `hypers`, either do an
    #   MH accept/reject step, or simply return.
    # For the MH weight, see
    # https://www.notion.so/chi-mit/MH-move-on-pixel-association-pixel-depth-18b01b4b8cde800d8184d343f84c03c0?pvs=4
    k1, k2, k3 = split(key, 3)
    datapoint = st.datapoints.value[datapoint_idx]
    masked_gaussians = get_relevant_gaussians_for_datapoint(datapoint_idx, st, hypers)
    proposed_xyz, gaussian_weights, meta_dict = jax.vmap(
        lambda key, gaussian: (
            propose_xyz_given_gaussian_and_score_dp_gaussian_assoc(
                key, datapoint, gaussian, st.matter.probs, hypers, c
            )
        )
    )(split(k1, len(masked_gaussians.value)), masked_gaussians.value)
    gaussian_weights = jnp.where(masked_gaussians.flag, gaussian_weights, -jnp.inf)  # type: ignore
    chosen_idx = genjax.categorical(gaussian_weights)(k2)

    # if not hypers.always_accept_assoc_depth_move:
    # TODO: add MH accept/reject logic here.

    datapoint = st.datapoints.value[datapoint_idx]
    return datapoint.replace(
        xyz=proposed_xyz[chosen_idx],
        gaussian_idx=masked_gaussians.value[chosen_idx].idx,
    ), meta_dict


def propose_xyz_given_gaussian_and_score_camera_xy(
    key,
    datapoint: Datapoint,
    gaussian: Gaussian,
    hypers: Hyperparams,
    c: LogConfig,
) -> tuple[FloatArray, FloatArray, dict]:
    f = hypers.intrinsics.fx
    camxy = datapoint.camera_xy
    proposed_xyz, weight = propose_xyz_and_estimate_p_of_image_coords(
        key, v=jnp.array([*camxy, f]), mu=gaussian.xyz, Sigma=gaussian.xyz_cov
    )
    return proposed_xyz, weight, {}


def propose_xyz_given_gaussian_and_score_dp_gaussian_assoc(
    key,
    datapoint: Datapoint,
    gaussian: Gaussian,
    gaussian_probs: jnp.ndarray,
    hypers: Hyperparams,
    c: LogConfig,
) -> tuple[jnp.ndarray, FloatArray, dict]:
    # Compute weights
    prior_weight = jnp.log(gaussian_probs[gaussian.idx])
    xyz_weight_if_depth_return = genjax.mv_normal.logpdf(
        datapoint.xyz, gaussian.xyz, gaussian.xyz_cov
    )
    rgb_weight = jax.vmap(
        lambda ch: genjax.normal.logpdf(
            datapoint.rgb[ch], gaussian.rgb[ch], jnp.sqrt(gaussian.rgb_vars[ch])
        )
    )(jnp.arange(3)).sum()

    if not hypers.repopulate_depth_nonreturns:
        weight = jnp.where(
            gaussian.has_extreme_value(),
            -jnp.inf,
            (prior_weight + xyz_weight_if_depth_return + rgb_weight),
        )
        return (
            datapoint.xyz,
            weight,
            {},
        )

    # Handle depth non-return
    is_depth_nonreturn = datapoint.obs.depth == 0
    xyz_if_nonreturn, xyz_weight_if_nonreturn, mdict = (
        propose_xyz_given_gaussian_and_score_camera_xy(
            key, datapoint, gaussian, hypers, c
        )
    )
    xyz = jnp.where(is_depth_nonreturn, xyz_if_nonreturn, datapoint.xyz)
    xyz_weight = jnp.where(
        is_depth_nonreturn, xyz_weight_if_nonreturn, xyz_weight_if_depth_return
    )

    # Return
    weight = jnp.where(
        gaussian.has_extreme_value(), -jnp.inf, (prior_weight + xyz_weight + rgb_weight)
    )
    weight = jnp.where(jnp.isnan(weight), -jnp.inf, weight)

    return (
        xyz,
        weight,
        {
            "prior_weight": prior_weight,
            "xyz_weight_if_depth_return": xyz_weight_if_depth_return,
            "rgb_weight": rgb_weight,
        }
        if c.log
        else {},
    )


### Tiling-related functionality ###


def update_tiling(
    key, st: CondorGMMState, hypers: Hyperparams, c: LogConfig
) -> tuple[CondorGMMState, Metadata]:
    new_tiling = st.matter.tiling.update_tiling(st.matter.gaussians, key=key)
    new_st = st.replace({"matter": {"tiling": new_tiling}})
    return wrap(new_st, c, "update_tiling")


def get_relevant_datapoints_for_gaussian(
    gaussian_idx: int, st: CondorGMMState, hypers: Hyperparams
) -> Mask[Datapoint]:
    possible_dp_idxs = st.matter.tiling.relevant_datapoints_for_gaussian(gaussian_idx)
    return Mask(
        st.datapoints.value[possible_dp_idxs.value],
        jnp.logical_and(
            jnp.logical_and(
                possible_dp_idxs.flag,  # type: ignore
                st.datapoints.flag[possible_dp_idxs.value],  # type: ignore
            ),
            st.datapoints.value[possible_dp_idxs.value].gaussian_idx == gaussian_idx,
        ),
    )


def get_relevant_gaussians_for_datapoint(
    datapoint_idx: int, st: CondorGMMState, hypers: Hyperparams
) -> Mask[Gaussian]:
    possible_gaussian_idxs = st.matter.tiling.relevant_gaussians_for_datapoint(
        datapoint_idx
    )
    return Mask(
        st.matter.gaussians[possible_gaussian_idxs.value], possible_gaussian_idxs.flag
    )


### Initialization functionality ###


# @jax.jit
def reinitialize_unobserved_gaussians(key, st, hypers) -> CondorGMMState:
    k1, k2 = split(key)
    isnt_duplicate, gaussians = _replicate_gaussians_with_assoc(k1, st, hypers)

    new_rgb_vars = jnp.where(
        isnt_duplicate[:, None], gaussians.rgb_vars, 9.0 * gaussians.rgb_vars
    )
    new_covs = jnp.where(
        isnt_duplicate[:, None, None], gaussians.xyz_cov, 9.0 * gaussians.xyz_cov
    )
    origin = jnp.where(isnt_duplicate, gaussians.origin, -1)
    n_frames_since_last_had_assoc = jnp.where(
        isnt_duplicate, gaussians.n_frames_since_last_had_assoc, -1
    )
    gaussians = gaussians.replace(
        rgb_vars=new_rgb_vars,
        xyz_cov=new_covs,
        origin=origin,
        n_frames_since_last_had_assoc=n_frames_since_last_had_assoc,
    )
    gaussians = widen_dists_for_unobserved_object_gaussians(st, gaussians, hypers)

    tiling = st.matter.tiling.update_tiling(gaussians, k2)
    return st.replace({"matter": {"gaussians": gaussians, "tiling": tiling}})


def _replicate_gaussians_with_assoc(
    key, st: CondorGMMState, hypers: Hyperparams
) -> tuple[jnp.ndarray, Gaussian]:
    dont_change = jnp.logical_or(
        st.gaussian_has_assoc_mask, jnp.logical_not(model.is_bkg_mask(st, hypers))
    )
    probs = jnp.where(
        jnp.logical_and(st.gaussian_has_assoc_mask, model.is_bkg_mask(st, hypers)),
        st.matter.probs,
        0.0,
    )
    probs = probs / probs.sum()
    new_gaussian_idxs = jax.vmap(
        lambda idx, key: jnp.where(
            dont_change[idx], idx, genjax.categorical(jnp.log(probs))(key)
        )
    )(jnp.arange(len(st.matter.gaussians)), split(key, len(st.matter.gaussians)))
    new_gaussians = st.matter.gaussians[new_gaussian_idxs].replace(
        idx=jnp.arange(len(st.matter.gaussians))
    )
    return dont_change, new_gaussians


def widen_dists_for_unobserved_object_gaussians(
    st: CondorGMMState, gaussians: Gaussian, hypers: Hyperparams
) -> Gaussian:
    if not isinstance(hypers.initial_scene, model.SingleKnownObjectSceneState):
        return gaussians

    def get_gaussian_params(idx):
        (
            target_rgb_mean,
            target_variance_for_rgb_mean,
            _,
            _,
        ) = model._get_object_gaussian_evolve_rgb_targets(gaussians[idx], hypers)
        do_change = jnp.logical_and(
            gaussians[idx].object_idx > 0, st.gaussian_has_assoc_mask[idx]
        )
        new_rgb = jnp.where(do_change, target_rgb_mean, gaussians[idx].rgb)
        # For gaussians where the uncertainty on the mean color has become
        # broad, we also increase the variance of the color distribution.
        new_rgb_vars = jnp.where(
            do_change, target_variance_for_rgb_mean, gaussians[idx].rgb_vars
        )

        alpha = model._get_alpha_for_evolve_object_gaussian_weight(
            gaussians[idx], st.matter.probs, hypers
        )
        mixture_weight = jnp.where(
            do_change, alpha * GAMMA_RATE_PARAMETER, gaussians[idx].mixture_weight
        )

        return new_rgb, new_rgb_vars, mixture_weight

    rgbs, rgb_vars, mixture_weights = jax.vmap(get_gaussian_params)(
        jnp.arange(len(gaussians))
    )

    gaussians = gaussians.replace(
        rgb=rgbs, rgb_vars=rgb_vars, mixture_weight=mixture_weights
    )

    return gaussians

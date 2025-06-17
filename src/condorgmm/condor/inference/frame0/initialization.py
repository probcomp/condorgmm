import jax
from jax.random import split
import jax.numpy as jnp
from genjax import Mask
from ...types import (
    Hyperparams,
    Observation,
    CondorGMMState,
    Gaussian,
    VisualMatter,
    Datapoint,
    BackgroundOnlySceneState,
)
from ...tiling import Tiling, GridTilingConfig, GridTiling, MonolithicTiling
import condorgmm.condor.model.model as model


def initialize_state_with_obs_and_tiling_and_stratified_assocs(
    key,
    observations: Observation,
    hypers: Hyperparams,
    given_datapoint_assignment: jnp.ndarray | None = None,
):
    k1, k2 = split(key)
    tiling, datapoint_to_gaussian_idx = _initialize_tiling_and_assocs(k1, hypers)
    if given_datapoint_assignment is not None:
        datapoint_to_gaussian_idx = given_datapoint_assignment
    return CondorGMMState(
        scene=hypers.initial_scene,
        matter=VisualMatter(
            hypers.initial_new_gaussian_prior_params,
            hypers.default_background_evolution_params,
            _placeholder_gaussians(hypers),
            tiling,
        ),
        datapoints=Mask(
            _initialize_datapoints(observations, datapoint_to_gaussian_idx, hypers),
            hypers.datapoint_mask,
        ),
    )


def _placeholder_gaussians(hypers):
    def bkg_gaussian(i):
        return Gaussian(
            idx=i,
            xyz=jnp.zeros(3, dtype=jnp.float32),
            xyz_cov=jnp.eye(3, dtype=jnp.float32),
            rgb=jnp.zeros(3, dtype=jnp.float32),
            rgb_vars=jnp.ones(3, dtype=jnp.float32),
            mixture_weight=jnp.array(1.0, dtype=jnp.float32),
            origin=jnp.array(-1, dtype=jnp.int32),
            object_idx=jnp.array(0, dtype=jnp.int32),
            n_frames_since_last_had_assoc=jnp.array(-1, dtype=jnp.int32),
        )

    def obj_gaussian(i):
        return model.get_object_model_gaussian_in_camera_frame(
            i, hypers.initial_scene, hypers
        )

    def gaussian(i):
        if isinstance(hypers.initial_scene, BackgroundOnlySceneState):
            return bkg_gaussian(i)
        return jax.tree.map(
            lambda x, y: jnp.where(model.gaussian_is_background(i, hypers), x, y),
            bkg_gaussian(i),
            obj_gaussian(i),
        )

    return jax.vmap(gaussian)(jnp.arange(hypers.n_gaussians))


def _initialize_datapoints(observations, datapoint_to_gaussian_idx, hypers):
    return jax.vmap(
        lambda idx: Datapoint.from_obs_det(
            observations[idx], datapoint_to_gaussian_idx[idx], hypers
        )
    )(jnp.arange(len(observations)))


def _initialize_tiling_and_assocs(key, hypers) -> tuple[Tiling, jnp.ndarray]:
    if hypers.use_monolithic_tiling:
        tiling = MonolithicTiling(hypers.n_gaussians, len(hypers.datapoint_mask))
        assoc = tiling.get_stratified_datapoint_assignment(key)
        return tiling, assoc

    config = GridTilingConfig(
        tile_size_x=hypers.tile_size_x,
        tile_size_y=hypers.tile_size_y,
        intrinsics=hypers.intrinsics,
        n_gaussians=hypers.n_gaussians,
        max_n_gaussians_per_tile=hypers.max_n_gaussians_per_tile,
    )
    tiling, assoc = GridTiling.get_initial_tiling_and_stratified_datapoint_assignment(
        config, key=key
    )
    return tiling, assoc

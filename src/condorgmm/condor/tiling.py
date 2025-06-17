import math
from typing import Self
import warnings
import jax
import jax.numpy as jnp
from genjax import Pytree, Mask
from genjax.typing import BoolArray, IntArray
from condorgmm.utils.jax import xyz_to_pixel_coordinates
from .types import Intrinsics, Tiling, MyPytree, Gaussian

@Pytree.dataclass
class MonolithicTiling(Tiling):
    n_gaussians: int = Pytree.static()
    n_datapoints: int = Pytree.static()

    def relevant_datapoints_for_gaussian(self, gaussian_idx: int) -> Mask:
        return Mask(
            jnp.arange(self.n_datapoints), jnp.ones(self.n_datapoints, dtype=bool)
        )

    def relevant_gaussians_for_datapoint(self, datapoint_idx: int) -> Mask:
        return Mask(
            jnp.arange(self.n_gaussians), jnp.ones(self.n_gaussians, dtype=bool)
        )

    def gaussian_is_relevant_for_datapoint(
        self, gaussian_idx: int, datapoint_idx: int
    ) -> BoolArray:
        return jnp.array(True, dtype=bool)

    def update_tiling(self, gaussians: jnp.ndarray, key):
        return self

    def get_stratified_datapoint_assignment(self, key):
        dp_to_gaussian = jax.random.randint(
            key, (self.n_datapoints,), 0, self.n_gaussians
        )
        gaussian_to_dp = jax.random.choice(
            key, self.n_datapoints, (self.n_gaussians,), replace=False
        )
        return dp_to_gaussian.at[gaussian_to_dp].set(jnp.arange(self.n_gaussians))


@Pytree.dataclass
class GridTilingConfig(MyPytree):
    intrinsics: Intrinsics
    tile_size_x: int = Pytree.static()  # The width of each tile in pixels
    tile_size_y: int = Pytree.static()  # The height of each tile in pixels
    n_gaussians: int = Pytree.static()
    max_n_gaussians_per_tile: int = Pytree.static()

    def __post_init__(self):
        if self.n_tiles_x > self.n_gaussians:
            warnings.warn(f)
        if self.n_gaussians > self.max_n_gaussians_per_tile * self.n_tiles:
            warnings.warn(f)

    @property
    def n_tiles_x(self):
        return math.ceil(self.intrinsics.image_width / self.tile_size_x)

    @property
    def n_tiles_y(self):
        return math.ceil(self.intrinsics.image_height / self.tile_size_y)

    @property
    def n_tiles(self):
        return self.n_tiles_x * self.n_tiles_y

    @property
    def n_datapoints(self):
        return self.intrinsics.image_width * self.intrinsics.image_height

    @property
    def all_tile_indices(
        self,
    ):  # Returns a (n_tiles, 2) array of integer [y, x] tile indices
        return jax.vmap(
            lambda flat_idx: jnp.array(
                [flat_idx // self.n_tiles_x, flat_idx % self.n_tiles_x],
                dtype=int,
            )
        )(jnp.arange(self.n_tiles))

    def pixel_coordinate_to_tile_index(
        self, pixel_y: float, pixel_x: float
    ) -> jax.Array:
        tile_y = jnp.array(pixel_y // self.tile_size_y, dtype=int)
        tile_x = jnp.array(pixel_x // self.tile_size_x, dtype=int)
        tile_y = jnp.maximum(0, jnp.minimum(tile_y, self.n_tiles_y - 1))
        tile_x = jnp.maximum(0, jnp.minimum(tile_x, self.n_tiles_x - 1))
        return jnp.array([tile_y, tile_x], dtype=int)

    def tile_index_to_datapoint_indices(self, tile_y: int, tile_x: int) -> jnp.ndarray:
        def datapoint_index(y_in_tile, x_in_tile):
            pixel_y_coord = tile_y * self.tile_size_y + y_in_tile
            pixel_x_coord = tile_x * self.tile_size_x + x_in_tile
            return pixel_y_coord * self.intrinsics.image_width + pixel_x_coord

        return jax.vmap(
            lambda y: jax.vmap(lambda x: datapoint_index(y, x))(
                jnp.arange(self.tile_size_x)
            )
        )(jnp.arange(self.tile_size_y)).reshape(-1)

    def datapoint_index_to_tile_index(self, datapoint_idx: int) -> tuple[int, int]:
        pixel_x = datapoint_idx % self.intrinsics.image_width
        pixel_y = datapoint_idx // self.intrinsics.image_width
        tile_x = pixel_x // self.tile_size_x
        tile_y = pixel_y // self.tile_size_y
        return (tile_y, tile_x)


@Pytree.dataclass
class GridTiling(Tiling):
    config: GridTilingConfig
    gaussian_to_tile: jnp.ndarray
    # ^ (n_gaussians, 2).  Stores [tile_y, tile_x] for each Gaussian.
    tile_to_gaussians: jnp.ndarray
    # ^ (n_tiles_y, n_tiles_x, max_n_gaussians_per_tile).  Stores
    # the indices of the Gaussians in the tile, and some -1s (empty slots).
    # May not store all Gaussians in the tile.

    ## Tiling interface methods ##

    def relevant_datapoints_for_gaussian(self, gaussian_idx: int) -> Mask:
        tile_y, tile_x = self.gaussian_to_tile[gaussian_idx]
        tile_ys = [tile_y - 1, tile_y, tile_y + 1]
        tile_xs = [tile_x - 1, tile_x, tile_x + 1]
        masked_vals = [
            self._datapoints_in_tile(ty, tx) for ty in tile_ys for tx in tile_xs
        ]
        return Mask(
            jnp.concatenate([x.value for x in masked_vals]),
            jnp.concatenate([x.flag for x in masked_vals]),  # type: ignore
        )

    def relevant_gaussians_for_datapoint(self, datapoint_idx: int) -> Mask:
        tile_y, tile_x = self.config.datapoint_index_to_tile_index(datapoint_idx)
        tile_ys = [tile_y - 1, tile_y, tile_y + 1]
        tile_xs = [tile_x - 1, tile_x, tile_x + 1]
        masked_vals = [
            self._gaussians_in_tile(ty, tx) for ty in tile_ys for tx in tile_xs
        ]
        return Mask(
            jnp.concatenate([x.value for x in masked_vals]),
            jnp.concatenate([x.flag for x in masked_vals]),  # type: ignore
        )

    def gaussian_is_relevant_for_datapoint(
        self, gaussian_idx: int, datapoint_idx: int
    ) -> BoolArray:
        gaussian_tile_y, gaussian_tile_x = self.gaussian_to_tile[gaussian_idx]
        datapoint_tile_y, datapoint_tile_x = self.config.datapoint_index_to_tile_index(
            datapoint_idx
        )
        y_in_range = jnp.abs(gaussian_tile_y - datapoint_tile_y) <= 1
        x_in_range = jnp.abs(gaussian_tile_x - datapoint_tile_x) <= 1
        return jnp.logical_and(y_in_range, x_in_range)

    def update_tiling(self, gaussians: Gaussian, key=jax.random.key(0)):
        return GridTiling.from_gaussian_means(self.config, gaussians.xyz, key=key)

    ## Constructors ##

    @classmethod
    def get_initial_tiling_and_stratified_datapoint_assignment(
        cls, config: GridTilingConfig, key=jax.random.key(0)
    ) -> tuple[Self, jnp.ndarray]:
        # Note that the rest of this method depends upon the details of how
        # this tiling is indexed.  If the tiling produced by `get_initial_tiling`
        # changes, this method will need to be updated as well.
        tiling = cls.get_initial_tiling(config)

        k1, k2 = jax.random.split(key)

        def choose_random_gaussian(key, dp_idx):
            tile_y, tile_x = config.datapoint_index_to_tile_index(dp_idx)
            gaussians_in_tile = tiling.tile_to_gaussians[tile_y, tile_x]
            logprobs = jnp.where(gaussians_in_tile >= 0, 0.0, -jnp.inf)
            idx = jax.random.categorical(key, logprobs)
            return gaussians_in_tile[idx]

        datapoint_to_gaussian = jax.vmap(choose_random_gaussian)(
            jax.random.split(k1, config.n_datapoints),
            jnp.arange(config.n_datapoints),
        )

        # NOTE: we could add logic here to more carefully stratify the datapoint
        # assignments, and guarantee that each Gaussian has at least one association.
        # However, with 64 pixels per tile & 8 Gaussians per tile, even without
        # stratification, there is only an 8 * (7/8)^8 =~ 0.0016 probability
        # that some Gaussian in the tile ends up with no datapoints.

        return tiling, datapoint_to_gaussian

    @classmethod
    def get_initial_tiling(cls, config: GridTilingConfig) -> Self:
        n_gaussians_per_tile = config.n_gaussians // config.n_tiles
        n_tiles_with_extra = config.n_gaussians % config.n_tiles

        ## Construct tile_to_gaussians ##

        # All these arrays have shape (n_tiles,).
        tile_to_n_gaussians = n_gaussians_per_tile * jnp.ones(config.n_tiles, dtype=int)
        tile_to_n_gaussians = tile_to_n_gaussians + (
            jnp.arange(config.n_tiles) < n_tiles_with_extra
        ).astype(int)
        tile_to_final_gaussian_idx = jnp.cumsum(tile_to_n_gaussians)
        tile_to_first_gaussian_idx = jnp.concatenate(
            [jnp.array([0]), tile_to_final_gaussian_idx[:-1]]
        )

        def value_for_slot_in_tile_array(
            slot_idx: int, first_gaussian_idx: int, final_gaussian_idx: int
        ) -> IntArray:
            idx_if_in_range = first_gaussian_idx + slot_idx
            n_utilized_slots = final_gaussian_idx - first_gaussian_idx
            return jnp.where(slot_idx < n_utilized_slots, idx_if_in_range, -1)

        # (n_tiles, max_n_gaussians_per_tile)
        tile_to_gaussians = jax.vmap(
            lambda first, final: jax.vmap(
                lambda slot_idx: value_for_slot_in_tile_array(slot_idx, first, final)
            )(jnp.arange(config.max_n_gaussians_per_tile))
        )(tile_to_first_gaussian_idx, tile_to_final_gaussian_idx)

        # (n_tiles_y, n_tiles_x, max_n_gaussians_per_tile)
        tile_to_gaussians = tile_to_gaussians.reshape(
            config.n_tiles_y, config.n_tiles_x, config.max_n_gaussians_per_tile
        )

        ## Construct gaussian_to_tile ##
        first_gaussian_idx_in_smaller_tiles = n_tiles_with_extra * (
            n_gaussians_per_tile + 1
        )

        def flat_tile_idx_for_gaussian(gaussian_idx):
            return jnp.where(
                gaussian_idx < first_gaussian_idx_in_smaller_tiles,
                gaussian_idx // (n_gaussians_per_tile + 1),
                n_tiles_with_extra
                + (gaussian_idx - first_gaussian_idx_in_smaller_tiles)
                // n_gaussians_per_tile,
            )

        gaussian_to_flat_tile = jax.vmap(flat_tile_idx_for_gaussian)(
            jnp.arange(config.n_gaussians)
        )
        gaussian_to_tile = config.all_tile_indices[gaussian_to_flat_tile]

        ## Return ##
        return cls(config, gaussian_to_tile, tile_to_gaussians)

    @classmethod
    def from_gaussian_means(
        cls,
        config: GridTilingConfig,
        means: jnp.ndarray,  # (n_gaussians, 3)
        key=jax.random.key(0),
    ) -> Self:
        intrinsics = config.intrinsics
        pixel_coords = xyz_to_pixel_coordinates(
            means,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
        )
        gaussian_to_tile = jax.vmap(config.pixel_coordinate_to_tile_index)(
            pixel_coords[:, 0],
            pixel_coords[:, 1],
        )
        tile_to_gaussian = cls._stochastic_invert_gaussian_to_tile(
            gaussian_to_tile, config, M=config.max_n_gaussians_per_tile, key=key
        )
        return cls(config, gaussian_to_tile, tile_to_gaussian)

    ## Utility methods ##

    def _datapoints_in_tile(self, tile_y, tile_x) -> Mask[jnp.ndarray]:
        indices = self.config.tile_index_to_datapoint_indices(tile_y, tile_x)
        y_in_bounds = jnp.logical_and(tile_y >= 0, tile_y < self.config.n_tiles_y)
        x_in_bounds = jnp.logical_and(tile_x >= 0, tile_x < self.config.n_tiles_x)
        return Mask(
            indices,
            jnp.where(
                jnp.logical_and(y_in_bounds, x_in_bounds),
                jnp.ones_like(indices, dtype=bool),
                jnp.zeros_like(indices, dtype=bool),
            ),
        )

    def _gaussians_in_tile(self, tile_y, tile_x) -> Mask[jnp.ndarray]:
        tile_gaussians = self.tile_to_gaussians[tile_y, tile_x]
        y_in_bounds = jnp.logical_and(tile_y >= 0, tile_y < self.config.n_tiles_y)
        x_in_bounds = jnp.logical_and(tile_x >= 0, tile_x < self.config.n_tiles_x)
        in_bounds = jnp.logical_and(y_in_bounds, x_in_bounds)
        return Mask(
            tile_gaussians,
            jnp.where(
                jnp.logical_and(in_bounds, tile_gaussians >= 0),
                jnp.ones_like(tile_gaussians, dtype=bool),
                jnp.zeros_like(tile_gaussians, dtype=bool),
            ),
        )

    @staticmethod
    def _stochastic_invert_gaussian_to_tile(
        gaussian_to_tile,
        config: GridTilingConfig,
        M,
        *,
        K=64,
        R=4,
        key=jax.random.key(0),
    ):
        n_tiles_y, n_tiles_x = config.n_tiles_y, config.n_tiles_x
        n_gaussians = gaussian_to_tile.shape[0]

        tile_to_gaussian_large = -jnp.ones((n_tiles_y, n_tiles_x, K), dtype=int)
        gaussian_to_R_idxs = jax.random.randint(key, (n_gaussians, R), 0, K)
        tile_to_gaussian_large = tile_to_gaussian_large.at[
            jnp.repeat(gaussian_to_tile[:, 0:1], R, axis=1),
            jnp.repeat(gaussian_to_tile[:, 1:2], R, axis=1),
            gaussian_to_R_idxs,
        ].set(jnp.arange(n_gaussians)[:, None])
        return jax.vmap(
            lambda a: jax.vmap(
                lambda idxs_large: GridTiling._compress_to_M(idxs_large, M)
            )(a)
        )(tile_to_gaussian_large)

    @staticmethod
    def _compress_to_M(idxs_large, M):
        @Pytree.dataclass
        class State(MyPytree):
            nextidx: int  # into tiles_M
            tiles_M: jnp.ndarray

        def kernel(state, next_incoming_idx):
            tiles_M = state.tiles_M.at[state.nextidx].set(next_incoming_idx)
            already_present = jnp.any(state.tiles_M == next_incoming_idx)
            next_idx = jnp.where(
                jnp.logical_or(next_incoming_idx == -1, already_present),
                state.nextidx,
                state.nextidx + 1,
            )
            return State(next_idx, tiles_M), None

        state0 = State(0, -jnp.ones(M, dtype=int))
        final_state, _ = jax.lax.scan(kernel, state0, idxs_large)
        return final_state.tiles_M

import jax
import jax.numpy as jnp
from condorgmm.utils.jax import unproject
from condorgmm.condor.types import Intrinsics
import condorgmm.condor.tiling as t


def test_grid_tiling_config():
    config = t.GridTilingConfig(
        tile_size_x=16,
        tile_size_y=16,
        intrinsics=Intrinsics(
            jnp.array(1.0),
            jnp.array(1.0),
            jnp.array(1.0),
            jnp.array(1.0),
            jnp.array(1.0),
            jnp.array(1.0),
            image_height=64,
            image_width=32,
        ),
        n_gaussians=16,
        max_n_gaussians_per_tile=16,
    )

    ### Basic tests ###
    assert config.n_tiles_y == 4
    assert config.n_tiles_x == 2
    assert config.n_tiles == 8
    # fmt: off
    assert jnp.all(
        config.all_tile_indices == jnp.array([
            [0, 0], [0, 1],
            [1, 0], [1, 1],
            [2, 0], [2, 1],
            [3, 0], [3, 1]
        ])
    )
    # fmt: on

    ### Test datapoint_index_to_tile_index ###
    def pix_to_tile_idx(y, x):  # type: ignore
        return config.datapoint_index_to_tile_index(y * 32 + x)

    assert pix_to_tile_idx(0, 0) == (0, 0)
    assert pix_to_tile_idx(8, 8) == (0, 0)
    assert pix_to_tile_idx(15, 15) == (0, 0)
    assert pix_to_tile_idx(16, 0) == (1, 0)
    assert pix_to_tile_idx(31, 31) == (1, 1)
    assert pix_to_tile_idx(32, 0) == (2, 0)
    assert pix_to_tile_idx(47, 0) == (2, 0)
    assert pix_to_tile_idx(48, 0) == (3, 0)
    assert pix_to_tile_idx(63, 0) == (3, 0)
    assert pix_to_tile_idx(0, 16) == (0, 1)
    assert pix_to_tile_idx(0, 31) == (0, 1)
    assert pix_to_tile_idx(47, 27) == (2, 1)

    ### Test tile_index_to_datapoint_indices ###
    def tile_idx_to_pix_idxs(y, x):
        return jax.vmap(lambda dp_idx: jnp.array([dp_idx // 32, dp_idx % 32]))(
            config.tile_index_to_datapoint_indices(y, x)
        )

    pix_idxs = tile_idx_to_pix_idxs(0, 0)

    def contains(a, b):
        return jnp.any(jnp.all(a == b, axis=1))

    assert contains(pix_idxs, jnp.array([0, 0]))
    assert contains(pix_idxs, jnp.array([0, 1]))
    assert contains(pix_idxs, jnp.array([1, 0]))
    assert contains(pix_idxs, jnp.array([8, 8]))
    assert contains(pix_idxs, jnp.array([15, 15]))
    assert tile_idx_to_pix_idxs(0, 0).shape[0] == 16 * 16

    pix_idxs = tile_idx_to_pix_idxs(1, 0)
    assert jnp.any(pix_idxs == jnp.array([16, 0]))
    assert jnp.any(pix_idxs == jnp.array([31, 0]))

    pix_idxs = tile_idx_to_pix_idxs(2, 1)
    assert jnp.any(pix_idxs == jnp.array([47, 27]))

    pix_idxs = tile_idx_to_pix_idxs(3, 1)
    assert jnp.any(pix_idxs == jnp.array([63, 31]))

    ### Test pixel_coordinate_to_tile_index ###
    def pix_to_tile_idx(y, x):
        arr = config.pixel_coordinate_to_tile_index(y, x)
        return (arr[0], arr[1])

    assert pix_to_tile_idx(0.5, 0.5) == (0, 0)
    assert pix_to_tile_idx(7.5, 7.5) == (0, 0)
    assert pix_to_tile_idx(15.5, 15.5) == (0, 0)
    assert pix_to_tile_idx(15.99, 15.99) == (0, 0)
    assert pix_to_tile_idx(16.0, 0.0) == (1, 0)
    assert pix_to_tile_idx(-100, -100) == (0, 0)
    assert pix_to_tile_idx(1000, 1000) == (3, 1)


class TestGridTiling:
    def _config(self):
        config = t.GridTilingConfig(
            tile_size_x=2,
            tile_size_y=2,
            intrinsics=Intrinsics(
                jnp.array(2.0),
                jnp.array(2.0),
                jnp.array(4.0),
                jnp.array(2.0),
                jnp.array(1e-5),
                jnp.array(1e5),
                image_height=8,
                image_width=4,
            ),
            n_gaussians=16,
            max_n_gaussians_per_tile=4,
        )
        return config

    def _tiling(self):
        return t.GridTiling.get_initial_tiling(self._config())

    def test_get_initial_tiling(self):
        tiling = self._tiling()
        # fmt: off
        assert jnp.all(tiling.gaussian_to_tile == jnp.array([
            [0, 0], [0, 0], [0, 1], [0, 1],
            [1, 0], [1, 0], [1, 1], [1, 1],
            [2, 0], [2, 0], [2, 1], [2, 1],
            [3, 0], [3, 0], [3, 1], [3, 1]
        ]))
        # fmt: on

        for y in range(4):
            for x in range(2):
                flatidx = y * 2 + x
                assert tiling.tile_to_gaussians[y, x].shape[0] == 4
                assert (tiling.tile_to_gaussians[y, x] >= 0).sum() == 2
                assert 2 * flatidx in tiling.tile_to_gaussians[y, x]
                assert 2 * flatidx + 1 in tiling.tile_to_gaussians[y, x]

    def test_datapoint_assignment_initialization(self):
        tiling, assoc = (
            t.GridTiling.get_initial_tiling_and_stratified_datapoint_assignment(
                self._config()
            )
        )
        for dp_idx, gaussian_idx in enumerate(assoc):
            y, x = tiling.gaussian_to_tile[gaussian_idx]
            y2, x2 = tiling.config.datapoint_index_to_tile_index(dp_idx)
            assert y == y2
            assert x == x2
        n_unique_gaussians = len(jnp.unique(assoc))
        assert n_unique_gaussians > self._config().n_gaussians * 0.8

    def test_from_gaussian_means(self):
        pixel_coords = jnp.array(
            [[y + 0.5, x + 0.5] for y in range(8) for x in range(4)]
        )
        depths = jnp.arange(32)
        i = self._config().intrinsics
        coords_3d = jax.vmap(
            lambda x, y, z: unproject(x, y, z, i.fx, i.fy, i.cx, i.cy)
        )(pixel_coords[:, 0], pixel_coords[:, 1], depths)
        tiling = t.GridTiling.from_gaussian_means(self._config(), coords_3d)
        expected_tile = jnp.array(
            [[y // 2, x // 2] for y in range(8) for x in range(4)]
        )
        assert jnp.all(tiling.gaussian_to_tile == expected_tile)
        for y in range(4):
            for x in range(2):
                assert (tiling.tile_to_gaussians[y, x] >= 0).sum() == 4
                for ty, tx in tiling.gaussian_to_tile[tiling.tile_to_gaussians[y, x]]:
                    assert ty == y
                    assert tx == x

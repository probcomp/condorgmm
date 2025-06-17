import numpy as np
from condorgmm import GMM


def test_gmm_construction():
    gmm = GMM(
        np.zeros((2, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        np.zeros((2, 3), dtype=int),
        np.ones((2, 3)),
        np.ones((2, 3)),
        np.array([0.9, 0.1]),
    )
    filtered = gmm[gmm.probs > 0.5]
    assert filtered.spatial_means.shape == (1, 3)

import warp as wp
from dataclasses import dataclass
import numpy as np
from .gmm_warp import GMM_Warp


@dataclass
class Hyperparams:
    outlier_probability: float
    outlier_volume: float
    window_half_width: int


DEFAULT_HYPERPARAMS = Hyperparams(
    outlier_probability=0.05,
    outlier_volume=1e10,
    window_half_width=5,
)


@dataclass
class State:
    gmm: GMM_Warp
    log_score_image: wp.array
    mask: wp.array
    hyperparams: Hyperparams
    gaussian_mask: wp.array | None


def initialize_state(
    gmm=None, frame=None, height=None, width=None, hyperparams=DEFAULT_HYPERPARAMS
):
    if frame is not None:
        height, width = frame.rgb.shape[:2]
    elif height is not None and width is not None:
        pass
    else:
        raise ValueError("Either frame or height and width must be provided")

    log_score_image = wp.zeros((height, width), dtype=wp.float32)
    full_mask = wp.array(np.ones((height, width)), dtype=wp.bool)

    return State(gmm, log_score_image, full_mask, hyperparams, None)

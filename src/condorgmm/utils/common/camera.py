import numpy as np


def xyz_from_depth_image(z, fx, fy, cx, cy):
    v, u = np.mgrid[: z.shape[0], : z.shape[1]]
    x = (u - cx + 0.5) / fx
    y = (v - cy + 0.5) / fy
    xyz = np.stack([x, y, np.ones_like(x)], dtype=z.dtype, axis=-1) * z[..., None]
    return xyz

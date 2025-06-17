import jax.numpy as jnp
import functools
import jax

## Re-exports
from .profile import *  # noqa:F403


@jax.jit
def xyz_from_depth_image(z, fx, fy, cx, cy):
    v, u = jnp.mgrid[: z.shape[0], : z.shape[1]]
    x = (u + 0.5 - cx) / fx
    y = (v + 0.5 - cy) / fy
    xyz = jnp.stack([x, y, jnp.ones_like(x)], axis=-1) * z[..., None]
    return xyz


@jax.jit
def unproject(pixy, pixx, z, fx, fy, cx, cy):
    x = (pixx - cx) / fx
    y = (pixy - cy) / fy
    return jnp.array([x, y, 1]) * z


@functools.partial(
    jnp.vectorize,
    signature="(3)->(2)",
    excluded=(
        1,
        2,
        3,
        4,
    ),
)
def xyz_to_pixel_coordinates(xyz, fx, fy, cx, cy):
    x = fx * xyz[0] / (xyz[2]) + cx
    y = fy * xyz[1] / (xyz[2]) + cy
    return jnp.array([y, x])


def camera_frame_points_to_pixel_indices(vertices_C, intrinsics, INVALID_IDX):
    projected_pixel_coordinates = jnp.rint(
        xyz_to_pixel_coordinates(
            vertices_C,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
        )
        - 0.5
    )
    image_height, image_width = (
        intrinsics.image_height,
        intrinsics.image_width,
    )

    # Filter out points the 0 depth or that don't hit the camera.
    is_valid = (
        jnp.all(projected_pixel_coordinates >= 0, axis=-1)
        & jnp.all(
            projected_pixel_coordinates < jnp.array([image_height, image_width]),
            axis=-1,
        )
        & jnp.all(~jnp.isnan(projected_pixel_coordinates), axis=-1)
    )
    projected_pixel_coordinates = jnp.where(
        is_valid[:, None], projected_pixel_coordinates, INVALID_IDX
    )

    projected_pixel_coordinates = projected_pixel_coordinates.astype(jnp.int32)
    return projected_pixel_coordinates


def get_normalized_log_probs(log_p):
    return log_p - jax.scipy.special.logsumexp(log_p)

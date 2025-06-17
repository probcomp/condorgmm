import torch
from gsplat import rasterization


def transform_matrix_from_posquat(posquat):
    pos = posquat[:3]
    quat = posquat[3:] / torch.norm(posquat[3:], dim=-1, keepdim=True)

    # Convert quaternion to rotation matrix
    r = torch.zeros((*quat.shape[:-1], 3, 3), device=quat.device)
    qx, qy, qz, qw = quat.unbind(-1)
    r[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    r[0, 1] = 2 * (qx * qy - qz * qw)
    r[0, 2] = 2 * (qx * qz + qy * qw)
    r[1, 0] = 2 * (qx * qy + qz * qw)
    r[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    r[1, 2] = 2 * (qy * qz - qx * qw)
    r[2, 0] = 2 * (qx * qz - qy * qw)
    r[2, 1] = 2 * (qy * qz + qx * qw)
    r[2, 2] = 1 - 2 * (qx * qx + qy * qy)

    # Create transformation matrix
    transform = torch.zeros((*pos.shape[:-1], 4, 4), device=pos.device)
    transform[:3, :3] = r
    transform[:3, 3] = pos
    transform[3, 3] = 1.0
    return transform


def transform_points(points, transform):
    points_homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    return (transform @ points_homogeneous.T).T[..., :3]


def render_rgbd(
    camera_posquat,
    posquat,
    means,
    quats,
    scales,
    opacities,
    rgbs,
    viewmat,
    K,
    width,
    height,
):
    # Convert position and quaternion to 4x4 transformation matrix
    camera_posquat_transform = transform_matrix_from_posquat(camera_posquat)
    camera_posquat_transform_inv = torch.inverse(camera_posquat_transform)

    posquat_transform = transform_matrix_from_posquat(posquat)

    # Transform means by homogeneous transform matrix
    means = transform_points(means, posquat_transform)
    means = transform_points(means, camera_posquat_transform_inv)

    rgb = rasterization(
        means, quats, scales, opacities, rgbs, viewmat, K, width, height, packed=False
    )[0][0]

    # Get depth
    colors_but_actually_depths = torch.zeros_like(rgbs)
    colors_but_actually_depths[..., 0] = means[..., 0]
    colors_but_actually_depths[..., 1] = 1.0
    tmp = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors_but_actually_depths,
        viewmat,
        K,
        width,
        height,
        packed=False,
    )[0][0]
    depth = tmp[..., 0]
    silhouette = tmp[..., 1]
    return rgb, depth, silhouette

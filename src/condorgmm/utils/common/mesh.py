import trimesh
import numpy as np


def sample_surface_points(mesh: trimesh.Trimesh, n_points: int):
    samples, _, colors = trimesh.sample.sample_surface(
        mesh, n_points, sample_color=True
    )
    xyz = np.array(samples)
    colors = np.array(colors)[..., :3]
    return xyz, colors

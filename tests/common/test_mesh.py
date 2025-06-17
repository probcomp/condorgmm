import condorgmm
import trimesh


def test_sample_surface_points():
    condorgmm.rr_init("sample mesh surface points")
    mesh_path = condorgmm.get_assets_path() / "bop/ycbv/models/obj_000001.ply"
    xyz, colors = condorgmm.utils.common.sample_surface_points(
        trimesh.load(mesh_path), 10000
    )
    condorgmm.rr_log_cloud(xyz, "samples", colors[:, :3])

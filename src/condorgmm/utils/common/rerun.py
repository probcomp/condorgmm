import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R
import condorgmm

# Util for tracking whether a blueprint has been logged
# to rerun.
_blueprint_logged = False


def get_blueprint_logged():
    global _blueprint_logged
    return _blueprint_logged


def set_blueprint_logged(val):
    global _blueprint_logged
    _blueprint_logged = val


def rr_init(name="demo"):
    global _blueprint_logged
    _blueprint_logged = False
    rr.init(name)
    rr.connect("127.0.0.1:8812")
    condorgmm.rr_set_time(0)


def rr_log_rgb(rgb, channel="rgb"):
    rr.log(channel, rr.Image(rgb[..., :3]))


def rr_log_depth(depth, channel="depth"):
    rr.log(channel, rr.DepthImage(depth * 1.0))


def rr_log_mask(mask, channel="mask"):
    rr.log(channel, rr.DepthImage(mask * 1.0))


def rr_log_rgbd(rgbd, channel="rgbd"):
    rr_log_rgb(
        rgbd[..., :3],
        channel + "/rgb",
    )
    rr_log_depth(rgbd[..., 3], channel + "/depth")


def rr_log_cloud(cloud, channel="cloud", colors=None):
    if colors is None:
        rr.log(channel, rr.Points3D(cloud.reshape(-1, 3)))
    else:
        rr.log(channel, rr.Points3D(cloud.reshape(-1, 3), colors=colors.reshape(-1, 3)))


def rr_log_frustum(pose, fx, fy, height, width, channel="frustum"):
    if isinstance(pose, condorgmm.Pose):
        posequat = pose.posquat
    else:
        posequat = pose
    rr.log(
        channel,
        rr.Pinhole(
            focal_length=[fx, fy],
            height=height,
            width=width,
        ),
    )
    rr.log(
        channel,
        rr.Transform3D(
            translation=posequat[:3],
            quaternion=posequat[3:],
        ),
    )


def rr_log_pose(pose, channel="pose", scale=0.1, radii=0.01):
    if isinstance(pose, condorgmm.Pose):
        posequat = pose.posquat
    else:
        posequat = pose
    position = posequat[:3]
    origins = np.tile(position[None, ...], (3, 1))
    colors = np.eye(3)
    rotation_matrix = R.from_quat(posequat[3:]).as_matrix()
    rr.log(
        channel,
        rr.Arrows3D(
            origins=origins,
            vectors=rotation_matrix.T * scale,
            radii=radii,
            colors=colors,
        ),
    )


def rr_set_time(t=0):
    rr.set_time_sequence("step", t)


def rr_log_frame(frame, channel="frame", camera_pose=None):
    rgb = frame.rgb / 255.0
    d = frame.depth
    fx, fy, cx, cy = frame.intrinsics
    xyz = condorgmm.xyz_from_depth_image(d, fx, fy, cx, cy)
    if camera_pose is not None:
        if not isinstance(camera_pose, condorgmm.Pose):
            camera_pose = condorgmm.Pose(camera_pose)
        xyz = camera_pose.apply(xyz)
    condorgmm.rr_log_cloud(xyz, f"{channel}/cloud", colors=rgb)
    condorgmm.rr_log_depth(d, f"{channel}/depth")
    condorgmm.rr_log_rgb(rgb, f"{channel}/rgb")


def rr_log_gmm(gmm, channel="gmm", fill_mode=None, size_scalar=1.0):
    # filter out very low probability gaussians
    gmm = gmm[gmm.probs > 0.1 / gmm.spatial_means.shape[0]]
    # log
    rr.log(
        channel,
        rr.Ellipsoids3D(
            centers=gmm.spatial_means,
            half_sizes=gmm.spatial_scales * size_scalar,
            quaternions=gmm.quats,
            colors=gmm.rgb_means / 255.0,
            fill_mode=fill_mode,
        ),
    )

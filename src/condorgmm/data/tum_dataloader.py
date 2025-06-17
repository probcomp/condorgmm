import imageio
import yaml
from .base_dataloading import Video, Frame
from condorgmm.utils.common import get_root_path
import os
import numpy as np


class TUMVideo(Video):
    SCENE_NAMES = [
        "freiburg1_desk",
        "freiburg1_desk2",
        "freiburg1_room",
        "freiburg2_xyz",
        "freiburg3_long_office_household",
    ]

    def __init__(
        self,
        scene,
    ):
        assert scene in self.SCENE_NAMES
        base_dir = get_root_path() / "assets/TUM_RGBD"

        self.input_folder = os.path.join(base_dir, f"rgbd_dataset_{scene}")

        self.color_paths, self.depth_paths, self.poses = self.load_data()
        self.num_images = len(self.color_paths)

        config_path = get_root_path() / "assets/configs/TUM" / f"{scene}.yaml"
        with open(config_path, "r") as f:
            self.config_dict = yaml.safe_load(f)

        self.camera_params = self.config_dict["camera_params"]
        self.fx, self.fy, self.cx, self.cy = (
            self.camera_params["fx"],
            self.camera_params["fy"],
            self.camera_params["cx"],
            self.camera_params["cy"],
        )
        self.png_depth_scale = self.camera_params["png_depth_scale"]

        self.intrinsics = np.array([self.fx, self.fy, self.cx, self.cy])

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_data(self):
        frame_rate = 32
        if os.path.isfile(os.path.join(self.input_folder, "groundtruth.txt")):
            pose_list = os.path.join(self.input_folder, "groundtruth.txt")
        elif os.path.isfile(os.path.join(self.input_folder, "pose.txt")):
            pose_list = os.path.join(self.input_folder, "pose.txt")

        image_list = os.path.join(self.input_folder, "rgb.txt")
        depth_list = os.path.join(self.input_folder, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        color_paths, depth_paths = [], []
        poses = []
        for ix in indicies:
            (i, j, k) = associations[ix]
            color_paths += [os.path.join(self.input_folder, image_data[i, 1])]
            depth_paths += [os.path.join(self.input_folder, depth_data[j, 1])]
            poses += [pose_vecs[k]]

        return color_paths, depth_paths, poses

    def __len__(self):
        return self.num_images

    def get_object_mesh_from_id(self, id):
        raise Exception("No known meshes in R3D files")

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        depth = (
            np.asarray(imageio.imread(depth_path), dtype=np.int64)
            / self.png_depth_scale
        )

        return Frame(
            rgb=color,
            depth=depth,
            masks=None,
            intrinsics=self.intrinsics,
            camera_pose=self.poses[index],
        )

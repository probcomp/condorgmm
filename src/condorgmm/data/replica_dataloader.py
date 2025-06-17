import imageio
import yaml
from .base_dataloading import Video, Frame
from condorgmm.utils.common import get_root_path
import os
import numpy as np
import glob
from natsort import natsorted
from condorgmm.utils.common.pose import Pose


class ReplicaVideo(Video):
    SCENE_NAMES = [
        "room0",
        "room1",
        "room2",
        "office0",
        "office1",
        "office2",
        "office3",
        "office4",
    ]

    def __init__(
        self,
        scene,
    ):
        base_dir = get_root_path() / "assets/replica/Replica"
        self.input_folder = os.path.join(base_dir, f"{scene}")
        self.pose_path = os.path.join(self.input_folder, "traj.txt")

        self.color_paths = natsorted(
            glob.glob(f"{self.input_folder}/results/frame*.jpg")
        )
        self.depth_paths = natsorted(
            glob.glob(f"{self.input_folder}/results/depth*.png")
        )

        self.num_images = len(self.color_paths)

        self.poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_images):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w_pose = Pose.from_matrix(c2w)
            self.poses.append(c2w_pose)
        config_path = get_root_path() / "assets/configs/replica.yaml"
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

    def __len__(self):
        return self.num_images

    def get_object_mesh_from_id(self, id):
        raise Exception("No known meshes in R3D files")

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = imageio.imread(color_path)
        depth = (
            np.asarray(imageio.imread(depth_path), dtype=np.int64)
            / self.png_depth_scale
        )

        return Frame(
            rgb=color,
            depth=depth,
            masks=None,
            intrinsics=self.intrinsics,
            camera_pose=self.poses[index].posquat,
        )

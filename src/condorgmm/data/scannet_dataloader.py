from .base_dataloading import Video, Frame
from condorgmm.utils.common import get_root_path
from condorgmm.utils.common.pose import Pose
import os
import glob
import numpy as np
from natsort import natsorted
import imageio
import cv2


class ScanNetVideo(Video):
    SCENE_NAMES = [
        "scene0000_00",
        "scene0001_00",
        "scene0002_00",
        "scene0003_00",
    ]

    def __init__(
        self,
        scene_name: str,
    ):
        super().__init__()

        base_dir = get_root_path() / "assets/scannet/scans"
        scene_directory = f"{scene_name}"
        self.scene_data_dir = os.path.join(base_dir, scene_directory)

        self.color_paths = natsorted(glob.glob(f"{self.scene_data_dir}/color/*.jpg"))
        self.depth_paths = natsorted(glob.glob(f"{self.scene_data_dir}/depth/*.png"))
        self.pose_paths = natsorted(glob.glob(f"{self.scene_data_dir}/pose/*.txt"))

        self.intrinsics = np.array([1169.621094, 1167.105103, 646.295044, 489.927032])
        self.num_images = len(self.color_paths)

    def __len__(self):
        return self.num_images

    def get_object_mesh_from_id(self, id):
        raise Exception("No known meshes in R3D files")

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        pose_path = self.pose_paths[index]

        color = np.asarray(imageio.imread(color_path), dtype=float)
        depth = (
            np.asarray(imageio.imread(depth_path), dtype=np.int64).astype(np.float32)
            / 1000.0
        )
        depth = cv2.resize(
            depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        pose = np.loadtxt(pose_path)

        return Frame(
            rgb=color,
            depth=depth,
            masks=None,
            intrinsics=self.intrinsics,
            camera_pose=Pose.from_matrix(pose).posquat,
        )

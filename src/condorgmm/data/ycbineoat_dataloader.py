import cv2
from .base_dataloading import Video, Frame
from condorgmm.utils.common import get_root_path
from condorgmm.utils.common.pose import Pose
import os
import glob
import numpy as np
import imageio
import trimesh

YCB_MODEL_NAMES = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
]


class YCBinEOATVideo(Video):
    SCENE_NAMES = [
        "bleach0",
        "bleach_hard_00_03_chaitanya",
        "cracker_box_reorient",
        "cracker_box_yalehand0",
        "mustard0",
        "mustard_easy_00_02",
        "sugar_box1",
        "sugar_box_yalehand0",
        "tomato_soup_can_yalehand0",
    ]

    def __init__(
        self,
        scene_name: str,
    ):
        super().__init__()
        base_dir = get_root_path() / "assets/ycbineoat"
        self.ycb_dir = get_root_path() / "assets/bop/ycbv/train_real"

        self.video_dir = os.path.join(base_dir, scene_name)

        self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
        self.depth_files = sorted(glob.glob(f"{self.video_dir}/depth/*.png"))

        K = np.loadtxt(f"{self.video_dir}/cam_K.txt").reshape(3, 3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.intrinsics = np.array([fx, fy, cx, cy])

        self.gt_pose_files = sorted(glob.glob(f"{self.video_dir}/annotated_poses/*"))

        self.num_images = len(self.color_files)

        self.scene_name_to_object = {
            "bleach0": "021_bleach_cleanser",
            "bleach_hard_00_03_chaitanya": "021_bleach_cleanser",
            "cracker_box_reorient": "003_cracker_box",
            "cracker_box_yalehand0": "003_cracker_box",
            "mustard0": "006_mustard_bottle",
            "mustard_easy_00_02": "006_mustard_bottle",
            "sugar_box1": "004_sugar_box",
            "sugar_box_yalehand0": "004_sugar_box",
            "tomato_soup_can_yalehand0": "005_tomato_soup_can",
        }

        self.object_id = YCB_MODEL_NAMES.index(self.scene_name_to_object[scene_name])

    def __len__(self):
        return self.num_images

    def get_object_name_from_id(self, id):
        return YCB_MODEL_NAMES[id]

    def get_object_mesh_from_id(self, id):
        mesh = trimesh.load(
            os.path.join(self.ycb_dir, f'../models/obj_{f"{id + 1}".rjust(6, "0")}.ply')
        )
        mesh.vertices *= 0.001
        return mesh

    def __getitem__(self, index):
        rgb = imageio.imread(self.color_files[index])[..., :3]

        depth = cv2.imread(self.depth_files[index], -1) / 1e3
        depth[(depth < 0.01)] = 0.0

        gt_pose = Pose.from_matrix(np.loadtxt(self.gt_pose_files[index]).reshape(4, 4))

        filepath = self.color_files[index].replace("rgb", "gt_mask")
        mask = cv2.imread(filepath, -1)
        if len(mask.shape) == 3:
            for c in range(3):
                if mask[..., c].sum() > 0:
                    mask = mask[..., c]
                    break

        return Frame(
            rgb=rgb,
            depth=depth,
            intrinsics=self.intrinsics,
            masks=mask[None, ...],
            camera_pose=Pose.identity().posquat,
            object_poses=gt_pose[None, :].posquat,
            object_ids=np.array([self.object_id]),
        )

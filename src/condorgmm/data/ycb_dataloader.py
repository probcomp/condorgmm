from .base_dataloading import Video, Frame
from condorgmm.utils.common import get_root_path
from condorgmm.utils.common.pose import Pose
import trimesh
import os
import glob
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from PIL import Image


def remove_zero_pad(img_id):
    for i, ch in enumerate(img_id):
        if ch != "0":
            return img_id[i:]


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


class YCBVVideo(Video):
    SCENE_NAMES = range(0, 20)
    YCB_MODEL_NAMES = YCB_MODEL_NAMES

    def __init__(
        self,
        scene_id: int,
        ycb_dir=None,
    ):
        super().__init__()

        if ycb_dir is None:
            self.ycb_dir = get_root_path() / "assets/bop/ycbv/train_real"
        else:
            self.ycb_dir = ycb_dir

        scene_id = str(scene_id).rjust(6, "0")
        self.scene_data_dir = os.path.join(
            self.ycb_dir, scene_id
        )  # depth, mask, mask_visib, rgb; scene_camera.json, scene_gt_info.json, scene_gt.json

        self.scene_rgb_images_dir = os.path.join(self.scene_data_dir, "rgb")
        self.num_images = int(
            os.path.basename(
                sorted(glob.glob(self.scene_rgb_images_dir + "/*.png"))[-1]
            ).split(".")[0]
        )

        self.scene_depth_images_dir = os.path.join(self.scene_data_dir, "depth")
        self.mask_visib_dir = os.path.join(self.scene_data_dir, "mask_visib")

        with open(
            os.path.join(self.scene_data_dir, "scene_camera.json")
        ) as scene_cam_data_json:
            self.scene_cam_data = json.load(scene_cam_data_json)

        with open(
            os.path.join(self.scene_data_dir, "scene_gt.json")
        ) as scene_imgs_gt_data_json:
            self.scene_imgs_gt_data = json.load(scene_imgs_gt_data_json)

    @classmethod
    def training_scene(cls, scene_id):
        return cls(scene_id, ycb_dir=get_root_path() / "assets/bop/ycbv/train_real")

    def __len__(self):
        return self.num_images

    def get_object_mesh_from_id(self, id):
        mesh = trimesh.load(
            os.path.join(
                self.scene_data_dir, f'../../models/obj_{f"{id + 1}".rjust(6, "0")}.ply'
            )
        )
        mesh.vertices *= 0.001
        return mesh

    def get_object_name_from_id(self, id):
        return YCB_MODEL_NAMES[id]

    def __getitem__(self, index):
        img_id = str(index + 1).rjust(6, "0")

        image_cam_data = self.scene_cam_data[remove_zero_pad(img_id)]
        cam_depth_scale = image_cam_data["depth_scale"]
        image_cam_data = {k: np.array(v) for k, v in image_cam_data.items()}

        cam_K = np.array(image_cam_data["cam_K"]).reshape(3, 3)
        cam_R_w2c = np.array(image_cam_data["cam_R_w2c"]).reshape(3, 3)
        cam_t_w2c = np.array(image_cam_data["cam_t_w2c"]).reshape(3, 1)
        cam_pose_w2c = np.vstack(
            [np.hstack([cam_R_w2c, cam_t_w2c]), np.array([0, 0, 0, 1])]
        )
        cam_pose = np.linalg.inv(cam_pose_w2c)
        cam_pose[:3, 3] /= 1000.0
        camera_intrinsics = np.array(
            [
                cam_K[0, 0],
                cam_K[1, 1],
                cam_K[0, 2],
                cam_K[1, 2],
            ]
        )
        cam_pose = Pose.from_matrix(cam_pose).posquat

        objects_gt_data = self.scene_imgs_gt_data[remove_zero_pad(img_id)]
        object_types = []
        object_poses = []
        for d in objects_gt_data:
            model_R = np.array(d["cam_R_m2c"]).reshape(3, 3)
            model_t = np.array(d["cam_t_m2c"]) / 1000.0
            obj_id = d["obj_id"] - 1
            obj_pose = np.concatenate([model_t, Rot.from_matrix(model_R).as_quat()])
            obj_pose = (
                Pose(cam_pose) @ Pose(obj_pose)
            ).posquat  # convert to world frame
            object_types.append(obj_id)
            object_poses.append(obj_pose)

        object_types = np.array(object_types)
        object_poses = np.array(object_poses)

        num_objects = len(object_types)
        mask_visib_image_paths = np.array(
            [
                os.path.join(self.mask_visib_dir, f"{img_id}_{obj_idx:06}.png")
                for obj_idx in range(num_objects)
            ]
        )
        mask_visib_images = np.stack(
            [
                np.array(Image.open(mask_path)) > 0
                for mask_path in mask_visib_image_paths
            ]
        )

        rgb_filename = os.path.join(self.scene_rgb_images_dir, f"{img_id}.png")
        depth_filename = os.path.join(self.scene_depth_images_dir, f"{img_id}.png")

        rgb = np.array(Image.open(rgb_filename), dtype=np.uint8)
        depth = np.array(Image.open(depth_filename)) * cam_depth_scale / 1000.0

        return Frame(
            rgb=rgb,
            depth=depth,
            masks=mask_visib_images,
            intrinsics=camera_intrinsics,
            camera_pose=cam_pose,
            object_poses=object_poses,
            object_ids=object_types,
        )


class YCBTestVideo(YCBVVideo):
    SCENE_NAMES = range(48, 60)
    YCB_MODEL_NAMES = YCB_MODEL_NAMES

    def __init__(
        self,
        scene_id: int,
    ):
        super().__init__(
            scene_id=scene_id, ycb_dir=get_root_path() / "assets/bop/ycbv/test"
        )

from .base_dataloading import Video, Frame
import os
import glob
import json
import numpy as np
import subprocess
from pathlib import Path
import liblzfse  # https://pypi.org/project/pyliblzfse/
from natsort import natsorted
import cv2


class R3DVideo(Video):
    def __init__(self, filename):
        super().__init__()

        self.r3d_filename = filename  # depth, mask, mask_visib, rgb; scene_camera.json, scene_gt_info.json, scene_gt.json
        r3d_path = Path(self.r3d_filename)
        subprocess.run([f"cp {r3d_path} /tmp/{r3d_path.name}.zip"], shell=True)
        subprocess.run(
            [f"unzip -qq -o /tmp/{r3d_path.name}.zip -d /tmp/{r3d_path.name}"],
            shell=True,
        )
        self.datapath = f"/tmp/{r3d_path.name}"

        self.metadata = json.load(open(os.path.join(self.datapath, "metadata"), "r"))

        # Camera intrinsics
        K = np.array(self.metadata["K"]).reshape((3, 3)).T
        K = K
        fx = K[0, 0]
        fy = K[1, 1]

        # # TODO(akristoffersen): The metadata dict comes with principle points,
        # # but caused errors in image coord indexing. Should update once that is fixed.
        # cx, cy = W / 2, H / 2
        cx, cy = K[0, 2], K[1, 2]

        scaling_factor = self.metadata["dw"] / self.metadata["w"]
        self.intrinsics = np.array(
            [
                fx * scaling_factor,
                fy * scaling_factor,
                cx * scaling_factor,
                cy * scaling_factor,
            ],
            dtype=np.float32,
        )

        self.poses = np.array(self.metadata["poses"])  # (N, 7)
        # NB: Record3D / scipy use "scalar-last" format quaternions (x y z w)
        # https://fzheng.me/2017/11/12/quaternion_conventions_en/
        positions = self.poses[..., 4:] * np.array([1, -1, -1])  # (N, 3)
        quaternions = self.poses[..., :4] * np.array([-1, 1, 1, -1])  # (N, 4)

        self.camera_posquats = np.concatenate([positions, quaternions], axis=1)

        self.color_paths = natsorted(
            glob.glob(os.path.join(self.datapath, "rgbd", "*.jpg"))
        )
        self.depth_paths = natsorted(
            glob.glob(os.path.join(self.datapath, "rgbd", "*.depth"))
        )

        self.num_images = len(self.color_paths)

    def __len__(self):
        return self.num_images

    def get_object_mesh_from_id(self, id):
        raise Exception("No known meshes in R3D files")

    def __getitem__(self, index):
        rgb_filename = self.color_paths[index]
        depth_filename = self.depth_paths[index]

        with open(depth_filename, "rb") as depth_fh:
            raw_bytes = depth_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            depth = np.frombuffer(decompressed_bytes, dtype=np.float32)

        rgb = cv2.imread(rgb_filename)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        is_landscape = rgb.shape[0] < rgb.shape[1]
        res = (192, 256) if is_landscape else (256, 192)
        depth = depth.reshape(res)  # For a LiDAR 3D Video
        depth = np.nan_to_num(depth, nan=0.0)

        rgb = cv2.resize(
            rgb, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        rgb = rgb.astype(np.uint8)

        return Frame(
            rgb=rgb,
            depth=depth,
            masks=None,
            intrinsics=self.intrinsics,
            camera_pose=self.camera_posquats[index],
        )

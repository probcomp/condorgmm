import dataclasses
import numpy as np
from typing import Optional
import warp as wp
import cv2


@dataclasses.dataclass
class Frame:
    rgb: np.ndarray | wp.array
    depth: np.ndarray | wp.array
    intrinsics: np.ndarray
    camera_pose: Optional[np.ndarray | wp.array] = None
    object_poses: Optional[np.ndarray | wp.array] = None
    object_ids: Optional[np.ndarray | wp.array] = None
    masks: Optional[np.ndarray | wp.array] = None

    @property
    def height(self):
        return self.rgb.shape[0]

    @property
    def width(self):
        return self.rgb.shape[1]

    @property
    def fx(self):
        return self.intrinsics[0]

    @property
    def fy(self):
        return self.intrinsics[1]

    @property
    def cx(self):
        return self.intrinsics[2]

    @property
    def cy(self):
        return self.intrinsics[3]

    def downscale(self, factor):
        fx, fy, cx, cy = self.intrinsics

        new_cx = cx / factor - 0.5 / factor + 0.5
        new_cy = cy / factor - 0.5 / factor + 0.5
        new_fx = fx / factor
        new_fy = fy / factor
        new_intrinsics = np.array([new_fx, new_fy, new_cx, new_cy])

        return Frame(
            rgb=self.rgb[::factor, ::factor],
            depth=self.depth[::factor, ::factor],
            intrinsics=new_intrinsics,
            camera_pose=self.camera_pose,
            object_poses=self.object_poses,
            object_ids=self.object_ids,
            masks=(None if self.masks is None else self.masks[:, ::factor, ::factor]),
        )
        
    def upscale(self, factor):
        fx, fy, cx, cy = self.intrinsics

        # Scale up the intrinsics
        new_cx = cx * factor - 0.5 * factor + 0.5
        new_cy = cy * factor - 0.5 * factor + 0.5
        new_fx = fx * factor
        new_fy = fy * factor
        new_intrinsics = np.array([new_fx, new_fy, new_cx, new_cy])

        # Use cv2 to resize the images
        new_rgb = cv2.resize(self.rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
        new_depth = cv2.resize(self.depth, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        
        # Handle masks if they exist
        new_masks = None
        if self.masks is not None:
            new_masks = np.stack([
                cv2.resize(mask * 1.0, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)  > 0.5
                for mask in self.masks
            ])

        return Frame(
            rgb=new_rgb,
            depth=new_depth,
            intrinsics=new_intrinsics,
            camera_pose=self.camera_pose,
            object_poses=self.object_poses,
            object_ids=self.object_ids,
            masks=new_masks,
        )

    def crop(self, miny, maxy, minx, maxx):
        fx, fy, cx, cy = self.intrinsics
        cx, cy = cx - minx, cy - miny
        return Frame(
            rgb=self.rgb[miny:maxy, minx:maxx],
            depth=self.depth[miny:maxy, minx:maxx],
            intrinsics=np.array([fx, fy, cx, cy]),
            camera_pose=self.camera_pose,
            object_poses=self.object_poses,
            object_ids=self.object_ids,
            masks=(None if self.masks is None else self.masks[:, miny:maxy, minx:maxx]),
        )

    def crop_to_fraction(self, x_fraction, y_fraction):
        h, w = self.height, self.width
        miny = int(h * (1 - y_fraction) / 2)
        maxy = int(h * (1 + y_fraction) / 2)
        minx = int(w * (1 - x_fraction) / 2)
        maxx = int(w * (1 + x_fraction) / 2)
        return self.crop(miny, maxy, minx, maxx)

    @staticmethod
    def dummy_data():
        h, w = 480, 640
        return Frame(
            rgb=np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
            depth=np.random.rand(h, w) * 10.0,
            intrinsics=np.array([1000.0, 1000.0, w / 2.0, h / 2.0]),
            camera_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )

    def as_warp(self):
        return Frame(
            rgb=wp.array(self.rgb, dtype=wp.vec3),
            depth=wp.array(self.depth, dtype=wp.float32),
            intrinsics=self.intrinsics,
            camera_pose=self.camera_pose,
            object_poses=self.object_poses,
            object_ids=self.object_ids,
            masks=self.masks,
        )

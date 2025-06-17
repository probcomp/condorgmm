import os
import subprocess
import numpy as np
import cv2
from pathlib import Path

from .base_dataloading import Video
from .frame import Frame
from ..utils.common import get_root_path
from dataclasses import dataclass


def get_cache_dir():
    path = get_root_path() / "cache" / "vda"
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory {path}")
    return path


@dataclass
class CameraSpec:
    sensor_width: float
    sensor_height: float
    focal_length: float

    def get_intrinsics(self, height: int, width: int):
        fy = self.focal_length * height / self.sensor_height
        fx = self.focal_length * width / self.sensor_width
        cx = width / 2
        cy = height / 2
        return np.array([fx, fy, cx, cy], dtype=np.float32)


KNOWN_CAMERAS = {
    "iphone_13_mini": CameraSpec(
        sensor_width=0.00674, sensor_height=0.00505, focal_length=0.026
    ),
    "short_focal_length": CameraSpec(
        sensor_width=0.00674, sensor_height=0.00505, focal_length=0.012
    ),
}


def depthmap_to_metric_depth(
    depthmap: np.ndarray, min_depth_meters: float, max_depth_meters: float
):
    midas_min = depthmap.min()
    midas_max = depthmap.max()
    midas_normalized = (depthmap - midas_min) / (midas_max - midas_min)
    midas_normalized_inverted = 1.0 - midas_normalized
    metric_depth = min_depth_meters + midas_normalized_inverted * (
        max_depth_meters - min_depth_meters
    )
    return metric_depth


class MP4DepthAnythingVideo(Video):
    def __init__(
        self,
        video_path: str,
        min_depth_meters: float,
        max_depth_meters: float,
        encoder: str = "vits",
        camera_type="iphone_13_mini",
    ):
        super().__init__()
        self.encoder = encoder
        self.camera_type = camera_type
        self.min_depth_meters = min_depth_meters
        self.max_depth_meters = max_depth_meters
        self.mp4_path = Path(video_path)
        if not self.mp4_path.exists():
            raise FileNotFoundError(f"File not found: {video_path}")

        # If input is .mov, convert to .mp4
        if self.mp4_path.suffix.lower() == ".mov":
            mp4_path = get_cache_dir() / f"{self.mp4_path.stem}.mp4"
            self._convert_mov_to_mp4(self.mp4_path, mp4_path)
            self.mp4_path = mp4_path

        # Create output directory for depth maps in the cache directory
        self.output_dir = get_cache_dir() / f"{self.mp4_path.stem}_depth"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if depth maps already exist
        depth_path = self.output_dir / f"{self.mp4_path.stem}_depths.npz"
        if not depth_path.exists():
            print(f"Depth maps not found at {depth_path}, generating them...")
            self._generate_depth_maps()
        else:
            print(f"Found existing depth maps at {depth_path}")

        # Load video frames
        self._load_frames()

    def _convert_mov_to_mp4(self, mov_path: Path, mp4_path: Path):
        if mp4_path.exists():
            print(
                f"Found existing MP4 file at {mp4_path}; using this and skipping MOV to MP4 conversion."
            )
            return

        print(f"Converting {mov_path} to {mp4_path}.")
        try:
            command = [
                "ffmpeg",
                "-i",
                str(mov_path),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                str(mp4_path),
            ]
            print("Running command:", " ".join(command))
            subprocess.run(command, check=True, capture_output=True, text=True)
            print("Conversion complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error converting MOV to MP4: {e}")
            print(f"Command output: {e.stdout}")
            print(f"Command error: {e.stderr}")
            raise e

    def _generate_depth_maps(self):
        # Change working directory to project root to ensure correct paths
        os.chdir(str(get_root_path()))

        # Remove MPLBACKEND from environment if present to avoid backend conflicts
        removed_backend = False
        if "MPLBACKEND" in os.environ:
            og_backend = os.environ.get("MPLBACKEND")
            del os.environ["MPLBACKEND"]
            removed_backend = True

        command = [
            "pixi",
            "run",
            "-e",
            "vda",
            "run",
            str(self.mp4_path),
            str(self.output_dir),
            self.encoder,
        ]
        print("Running command:", " ".join(command))
        err = None
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            print(f"Command output: {e.stdout}")
            print(f"Command error: {e.stderr}")
            print()
            print("ERROR when trying to run Video Depth Anything.")
            print(
                "Make sure you have run `pixi run -e vda setup` before using MP4DepthAnythingVideo."
            )
            err = e
        finally:
            if removed_backend:
                os.environ["MPLBACKEND"] = og_backend

            if err:
                raise err

    def _load_frames(self):
        # Open video file
        cap = cv2.VideoCapture(str(self.mp4_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.mp4_path}")

        # Get video properties
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load all frames
        self.rgb_frames = []

        for i in range(self.num_frames):
            # Read RGB frame
            ret, frame = cap.read()
            if not ret:
                break
            # Swap red and blue channels
            frame = frame[:, :, [2, 1, 0]]
            self.rgb_frames.append(frame)

        cap.release()

        # Load all depth maps from the single .npz file
        depth_path = self.output_dir / f"{self.mp4_path.stem}_depths.npz"
        print(f"Loading depth maps from: {depth_path}")
        if depth_path.exists():
            depth_data = np.load(str(depth_path))
            self.depth_frames = depth_data[
                "depths"
            ]  # Assuming the depth array is stored with key 'depths'
        else:
            raise FileNotFoundError(f"Depth maps not found: {depth_path}")

        depth_shape = self.depth_frames[0].shape
        self.intrinsics = KNOWN_CAMERAS[self.camera_type].get_intrinsics(
            depth_shape[0], depth_shape[1]
        )

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> Frame:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Frame index {idx} out of range [0, {len(self)-1}]")

        rgb = self.rgb_frames[idx]
        depth = self.depth_frames[idx]
        depth = depthmap_to_metric_depth(
            depth, self.min_depth_meters, self.max_depth_meters
        )

        # Shrink RGB to match depth map size if necessary
        if rgb.shape[:2] != depth.shape:
            rgb = cv2.resize(
                rgb,
                dsize=(depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            rgb = rgb.astype(np.uint8)

        return Frame(
            rgb=rgb,
            depth=depth,
            intrinsics=self.intrinsics,
        )

    def get_object_mesh_from_id(self, id: int):
        raise NotImplementedError("This video does not contain object meshes")

    def get_object_name_from_id(self, id: int) -> str:
        raise NotImplementedError("This video does not contain objects")

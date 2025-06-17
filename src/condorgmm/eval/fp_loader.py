# Utility scripts to load precomputed pose tracking results from FoundationPose
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

import condorgmm

FP_RESULTS_ROOT_DIR = (
    condorgmm.get_assets_path() / "condorgmm_bucket/foundation_pose_tracking_results"
)
FP_YCBV_RESULTS_ROOT_DIR = FP_RESULTS_ROOT_DIR / "ycbv"


class _Config(NamedTuple):
    frame_rate: int
    split: str
    # unless otherwise specified, we will use the ground truth to initialize the
    # first frame of the pose tracking
    init_from_gt: bool = True


PRECOMPUTED_FP_RESULTS = {
    _Config(frame_rate=1, split="test"): (
        FP_YCBV_RESULTS_ROOT_DIR / "2024-10-03-original-frames-gt-init"
    ),
    _Config(frame_rate=50, split="test"): (
        FP_YCBV_RESULTS_ROOT_DIR / "2024-07-11-every-50-frames-gt-init"
    ),
    _Config(frame_rate=1, split="train_real"): (
        FP_YCBV_RESULTS_ROOT_DIR / "2024-09-27-original-frames-gt-init-training-set"
    ),
    _Config(frame_rate=50, split="train_real"): (
        FP_YCBV_RESULTS_ROOT_DIR / "2024-09-26-every-50-frames-gt-init-training-set"
    ),
}


class YCBVTrackingResultLoader:
    def __init__(self, frame_rate: int, split: str):
        config_key = _Config(frame_rate=frame_rate, split=split)
        result_dir = PRECOMPUTED_FP_RESULTS.get(config_key)

        if result_dir is None or not result_dir.exists():
            raise ValueError(
                f"Cannot find the precomputed results for {config_key}. Did you"
                " pull from b3d_bucket? (Hint: try `python scripts/bucket_utils/pull.py`)"
            )
        self.result_dir = result_dir

    @property
    def metrics_dir(self) -> Path:
        return self.result_dir / "metrics"

    def get_poses(self, test_scene_id: int, object_id: int) -> np.ndarray:
        filename = self.result_dir / str(test_scene_id) / f"object_{object_id}.npy"
        return np.load(filename)

    def get_gt_poses(self, test_scene_id: int, object_id: int) -> np.ndarray:
        filename = (
            self.result_dir
            / "gt_poses"
            / str(test_scene_id)
            / f"object_{object_id}.npy"
        )
        return np.load(filename)

    def get_metrics(
        self, test_scene_id: int, object_id: int
    ) -> dict[str, dict[str, list[float]]]:
        filename = (
            self.metrics_dir
            / f"SCENE_{test_scene_id}_OBJECT_INDEX_{object_id}_METRICS.json"
        )
        with open(filename, "r") as f:
            return json.load(f)

    def get_scene_ids(self) -> list[int]:
        return sorted(
            [
                int(test_scene.name)
                for test_scene in self.result_dir.iterdir()
                if test_scene.name.isdigit()
            ]
        )

    def get_object_ids(self, test_scene_id: int) -> list[int]:
        scene_dir = self.result_dir / str(test_scene_id)
        prefix_length = len("object_")
        return sorted(
            [int(object_id.stem[prefix_length:]) for object_id in scene_dir.iterdir()]
        )

    def get_dataframe(self, test_scene_id: int) -> pd.DataFrame:
        filename = (
            self.result_dir
            / "dataframes"
            / f"scene_{test_scene_id}_object_pose_tracking_results.pkl"
        )
        return pd.read_pickle(filename)

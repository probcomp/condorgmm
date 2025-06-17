import numpy as np
from dataclasses import dataclass
from .pose import Pose


@dataclass
class GMM:
    spatial_means: np.ndarray  # (N, 3)
    quats: np.ndarray  # (N, 4), xyzw
    rgb_means: np.ndarray  # (N, 3)
    spatial_scales: np.ndarray  # (N, 3)
    rgb_scales: np.ndarray  # (N, 3)
    probs: np.ndarray  # (N, )

    def __getitem__(self, idx):
        return GMM(
            spatial_means=self.spatial_means[idx],
            quats=self.quats[idx],
            rgb_means=self.rgb_means[idx],
            spatial_scales=self.spatial_scales[idx],
            rgb_scales=self.rgb_scales[idx],
            probs=self.probs[idx],
        )

    def transform_by(self, transform: Pose):
        gaussian_poses = Pose.from_pos_and_quat(self.spatial_means, self.quats)
        gaussian_poses = transform @ gaussian_poses

        return GMM(
            spatial_means=gaussian_poses.pos,
            quats=gaussian_poses.xyzw,
            rgb_means=self.rgb_means,
            spatial_scales=self.spatial_scales,
            rgb_scales=self.rgb_scales,
            probs=self.probs,
        )

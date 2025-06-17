import numpy as np
import condorgmm.warp_gmm as warp_gmm
import condorgmm
import condorgmm.data
from tqdm import tqdm


def test_warp_gmm_EM():
    condorgmm.rr_init("test_gmm_fitting_with_EM")

    scene = "office1"
    video = condorgmm.data.ReplicaVideo(scene)
    max_T_local = 10
    frames = [video[i] for i in tqdm(range(0, max_T_local, 1))]
    frame = frames[0]

    STRIDE = 30
    spatial_means = condorgmm.xyz_from_depth_image(
        frame.depth.astype(np.float32), *frame.intrinsics
    )[::STRIDE, ::STRIDE].reshape(-1, 3)
    rgb_means = frame.rgb[::STRIDE, ::STRIDE].reshape(-1, 3).astype(np.float32)

    scales_xyz = np.log(
        np.tile(
            np.array([0.01, 0.01, 0.01], dtype=np.float32), (spatial_means.shape[0], 1)
        )
    )
    scales_rgb = np.log(
        np.tile(np.array([5.1, 5.1, 5.1], dtype=np.float32), (rgb_means.shape[0], 1))
    )

    gmm = warp_gmm.gmm_warp_from_numpy(spatial_means, rgb_means, scales_xyz, scales_rgb)

    warp_gmm_state = warp_gmm.initialize_state(gmm, frame)
    warp_gmm_state.hyperparams.window_half_width = 10
    frame_warp = frame.as_warp()

    condorgmm.rr_init("test_EM")

    frame_warp = frame.as_warp()
    condorgmm.rr_set_time(-1)
    warp_gmm.rr_log_gmm_warp(warp_gmm_state.gmm)

    for t in range(10):
        condorgmm.rr_set_time(t)
        warp_gmm.warp_gmm_EM_step(frame_warp, warp_gmm_state)
        warp_gmm.rr_log_gmm_warp(warp_gmm_state.gmm)

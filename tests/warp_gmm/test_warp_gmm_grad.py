import numpy as np
import warp as wp
import condorgmm.warp_gmm as warp_gmm
from tqdm import tqdm
import condorgmm
import matplotlib.pyplot as plt
from warp.optim import Adam


def test_gradient():
    scene = "office1"
    video = condorgmm.data.ReplicaVideo(scene)
    max_T_local = 10
    frames = [video[i] for i in tqdm(range(0, max_T_local, 1))]
    frame = frames[0]

    initial_camera_pose = condorgmm.Pose(frame.camera_pose)

    STRIDE = 3
    spatial_means = condorgmm.xyz_from_depth_image(
        frame.depth.astype(np.float32), *frame.intrinsics
    )[::STRIDE, ::STRIDE].reshape(-1, 3)
    spatial_means = initial_camera_pose.apply(spatial_means).astype(np.float32)
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
    frame_warp = frame.as_warp()

    warp_gmm_state.camera_posquat = wp.array(
        initial_camera_pose.posquat.astype(np.float32)
        + np.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    )

    warp_gmm_state.camera_posquat.requires_grad = True

    optimizer = Adam([warp_gmm_state.camera_posquat], lr=1e-3)

    tape = wp.Tape()
    with tape:
        warp_gmm.warp_gmm_forward(
            frame_warp,
            warp_gmm_state,
        )

    tape.backward(grads={warp_gmm_state.log_score_image: warp_gmm_state.backward})

    print("before ", warp_gmm_state.camera_posquat.numpy())
    print("grad ", warp_gmm_state.camera_posquat.grad.numpy())
    optimizer.step([warp_gmm_state.camera_posquat.grad])
    print("after ", warp_gmm_state.camera_posquat.numpy())

    plt.matshow(warp_gmm_state.log_score_image.numpy())
    plt.matshow(warp_gmm_state.log_score_image.numpy())
    plt.colorbar()
    plt.savefig("test_gradient_log_score_image.png")

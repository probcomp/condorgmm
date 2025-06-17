import condorgmm
import condorgmm.warp_gmm as warp_gmm
import condorgmm.eval.metrics
import condorgmm.data
from tqdm import tqdm
import condorgmm.camera_tracking
import warp as wp
import numpy as np
import condorgmm.warp_gmm.kernels
import fire
from condorgmm import Pose
import condorgmm
import condorgmm.utils.common.rerun
import datetime as dt


def run_runtime_accuracy(experiment_name=None):
    num_timesteps = 100

    condorgmm.rr_init("runtime accuracy")

    scene_id = 3
    object_index = 2
    video = condorgmm.data.YCBVVideo(scene_id)
    frames = video.load_frames(range(num_timesteps))
    frames_warp = [frame.as_warp() for frame in frames]

    start_time = 0

    condorgmm.rr_init(f"object tracking scene {scene_id} object {object_index}")
    object_id = frames[start_time].object_ids[object_index]
    object_name = video.get_object_name_from_id(object_id)
    object_mesh = video.get_object_mesh_from_id(object_id)

    results_df = condorgmm.eval.metrics.create_empty_results_dataframe()
    results_df["fps"] = None
    results_df["num_gaussians"] = None

    print(f"Object name: {object_name}")

    if experiment_name is None:
        experiment_name = ""
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = f"{experiment_name}_{timestamp}"
    print(f"Experiment name: {experiment_name}")

    results_dir = (
        condorgmm.get_root_path()
        / "results"
        / f"runtime_accuracy_{scene_id}_{object_index}_{experiment_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    for STRIDE in [40, 35, 30, 25, 20, 15, 13, 10, 8, 5, 3, 2, 1]:
        frame = frames[start_time]
        mask = frame.masks[object_index] * frame.depth > 0.001
        camera_pose = condorgmm.Pose(frame.camera_pose)
        object_pose = condorgmm.Pose(frame.object_poses[object_index])

        mask_strided = mask[::STRIDE, ::STRIDE]
        xyz = condorgmm.xyz_from_depth_image(frame.depth, *frame.intrinsics)
        xyz = camera_pose.apply(xyz)
        xyz_strided = xyz[::STRIDE, ::STRIDE]
        rgb_strided = frame.rgb[::STRIDE, ::STRIDE]
        spatial_means = xyz_strided[mask_strided]
        rgb_means = rgb_strided[mask_strided]

        warp_gmm_state = warp_gmm.initialize_state(frame=frame)
        warp_gmm_state.hyperparams.window_half_width = 3

        gmm = warp_gmm.gmm_warp_from_numpy(
            object_pose.inv().apply(spatial_means).astype(np.float32),
            rgb_means.astype(np.float32),
        )
        warp_gmm_state.gmm = gmm
        warp_gmm_state.gmm.object_posquats = wp.array(
            object_pose.posquat[None, ...], dtype=wp.float32
        )
        warp_gmm_state.gmm.camera_posquat = wp.array(
            camera_pose.posquat, dtype=wp.float32
        )
        frame_warp = frame.as_warp()
        warp_gmm.warp_gmm_forward(frame_warp, warp_gmm_state)
        # log_score_image_before = warp_gmm_state.log_score_image.numpy()

        full_mask = wp.array(
            np.ones((frame.height, frame.width), dtype=wp.bool), dtype=wp.bool
        )
        partial_mask = wp.array(mask, dtype=wp.bool)

        warp_gmm_state.mask = partial_mask
        for _ in range(10):
            warp_gmm.warp_gmm_EM_step(frame_warp, warp_gmm_state)
        warp_gmm_state.mask = full_mask

        warp_gmm.warp_gmm_forward(frame_warp, warp_gmm_state)
        log_score_image_after = warp_gmm_state.log_score_image.numpy()

        warp_gmm.rr_log_gmm_warp(warp_gmm_state.gmm)
        condorgmm.rr_log_frame(frame)
        condorgmm.rr_log_depth(log_score_image_after, channel="frame/log_score_image")

        print("Number of gaussians : ", warp_gmm_state.gmm.spatial_means.shape)

        position_lr = 0.003
        quat_lr = 0.002
        learning_rates = wp.array(
            [position_lr, position_lr, position_lr, quat_lr, quat_lr, quat_lr, quat_lr],
            dtype=wp.float32,
        )

        warp_gmm_state.gmm.object_posquats.requires_grad = True

        camera_poses = []
        object_poses = []
        pbar = tqdm(range(num_timesteps))
        for t in pbar:
            condorgmm.rr_set_time(t)
            frame = frames[t]
            frame_warp = frames_warp[t]
            warp_gmm.optimize_params(
                [
                    warp_gmm_state.gmm.object_posquats,
                ],
                frame_warp,
                warp_gmm_state,
                10,
                learning_rates,
                storing_stuff=False,
            )
            camera_pose_np = condorgmm.Pose(warp_gmm_state.gmm.camera_posquat.numpy())
            object_pose_np = condorgmm.Pose(warp_gmm_state.gmm.object_posquats.numpy()[0])
            camera_poses.append(camera_pose_np)
            object_poses.append(object_pose_np)
            condorgmm.rr_log_posquat(
                (camera_pose_np.inv() @ object_pose_np).posquat,
                "object_pose_in_camera_frame",
            )
            condorgmm.rr_log_posquat(
                Pose(frame.camera_pose).inv() @ Pose(frame.object_poses[object_index]),
                "object_pose_in_camera_frame_gt",
            )

        # get fps from pbar
        fps = num_timesteps / pbar.format_dict["elapsed"]

        gt_poses = [
            Pose(frame.camera_pose).inv() @ Pose(frame.object_poses[object_index])
            for frame in frames
        ]
        predicted_poses = [
            camera_pose_np.inv() @ object_pose_np
            for camera_pose_np, object_pose_np in zip(camera_poses, object_poses)
        ]

        condorgmm.eval.metrics.add_object_tracking_metrics_to_results_dataframe(
            results_df,
            scene_id,
            "condorgmm",
            object_name,
            predicted_poses,
            gt_poses,
            object_mesh.vertices,
            other_info={
                "fps": fps,
                "num_gaussians": warp_gmm_state.gmm.spatial_means.shape[0],
            },
        )

        aggregated_df = results_df.groupby(
            ["metric", "num_gaussians", "fps", "object", "method"]
        )["value"].apply(lambda x: x.mean())

        print(aggregated_df)
        del warp_gmm_state

    results_df.to_pickle(results_dir / "runtime_accuracy_results.pkl")


if __name__ == "__main__":
    fire.Fire(run_runtime_accuracy)

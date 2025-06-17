import os
import condorgmm
import fire
import numpy as np
import matplotlib.pyplot as plt
import torch
import condorgmm.eval.metrics
import condorgmm.data
from tqdm import tqdm
from condorgmm.ng.torch_utils import render_rgbd
import datetime as dt


os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{os.environ['LD_LIBRARY_PATH']}"
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"
os.environ["PATH"]
device = torch.device("cuda:0")


def run_runtime_accuracy(experiment_name=None):
    num_timesteps = 100

    # scene_id, object_index = 3, 1
    scene_id, object_index = 6, 3
    scene_id, object_index = 3, 2
    video = condorgmm.data.YCBVVideo(scene_id)
    frames = video.load_frames(range(num_timesteps))

    frame0 = frames[0]
    fx, fy, cx, cy = frame0.intrinsics
    height, width = frame0.depth.shape

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    object_id = frames[0].object_ids[object_index]
    object_name = video.get_object_name_from_id(object_id)
    object_mesh = video.get_object_mesh_from_id(object_id)

    # Plot frame 0
    axes[0].imshow(frames[0].rgb * frames[0].masks[object_index][..., None])
    axes[0].set_title("Frame 0")
    axes[0].axis("off")

    # Plot frame 100
    axes[1].imshow(frames[-1].rgb * frames[-1].masks[object_index][..., None])
    axes[1].set_title("Frame 100")
    axes[1].axis("off")

    plt.tight_layout()

    viewmat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    K = torch.tensor(
        [
            [frame0.fx, 0, frame0.width / 2],
            [0, frame0.fy, frame0.height / 2],
            [0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )

    results_df = condorgmm.eval.metrics.create_empty_results_dataframe()
    results_df["fps"] = None
    results_df["num_gaussians"] = None

    for num_gaussians in [10, 20, 50, 121, 187, 491, 1370, 3096, 12337]:
        object_mask = frame0.masks[object_index]
        object_mask = frame0.masks[object_index]

        camera_pose = condorgmm.Pose(frame0.camera_pose)
        object_pose = condorgmm.Pose(frame0.object_poses[object_index])

        sampled_indices = np.random.choice(object_mask.sum(), num_gaussians)

        rgb_means_np = frame0.rgb[object_mask][sampled_indices].astype(np.float32)
        spatial_means_np = camera_pose.apply(
            condorgmm.utils.common.xyz_from_depth_image(frame0.depth, fx, fy, cx, cy)[
                object_mask
            ][sampled_indices].astype(np.float32)
        )

        spatial_means_np = object_pose.inv().apply(spatial_means_np)

        means = torch.tensor(
            spatial_means_np, device=device, requires_grad=True, dtype=torch.float32
        )
        rgbs = torch.tensor(
            rgb_means_np, device=device, requires_grad=True, dtype=torch.float32
        )

        quats = torch.randn(num_gaussians, 4, device=device, requires_grad=True)
        quats = torch.tensor(
            np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (num_gaussians, 1)),
            device=device,
            requires_grad=True,
            dtype=torch.float32,
        )
        scales = torch.tensor(
            torch.log(torch.randn(num_gaussians, 3) * 0.001),
            device=device,
            requires_grad=True,
            dtype=torch.float32,
        )
        opacities = torch.tensor(
            np.ones(num_gaussians) * 4.0,
            device=device,
            requires_grad=True,
            dtype=torch.float32,
        )

        # Get target image
        target_img = torch.tensor(
            frame0.rgb * object_mask[..., None], device=device
        ).float()
        target_depth = torch.tensor(frame0.depth * object_mask, device=device).float()

        scales.requires_grad_ = True
        opacities.requires_grad_ = True

        posquat = torch.tensor(object_pose.posquat, device=device, requires_grad=True)
        camera_posquat = torch.tensor(
            camera_pose.posquat, device=device, requires_grad=True
        )

        # Setup optimizer
        optimizer = torch.optim.Adam([means, rgbs, opacities, scales], lr=6e-4)
        n_steps = 1000
        first_frame = None
        pbar = tqdm(range(n_steps))
        for step in pbar:
            optimizer.zero_grad()

            # Forward pass
            rendered_rgb, rendered_depth, rendered_silhouette = render_rgbd(
                camera_posquat,
                posquat,
                means,
                quats,
                torch.exp(scales),
                torch.sigmoid(opacities),
                rgbs,
                viewmat[None],
                K[None],
                frame0.width,
                frame0.height,
            )
            if first_frame is None:
                first_frame = rendered_rgb

            # Compute losses
            # 1. RGB loss within mask
            rgb_loss = (torch.abs(rendered_rgb - (target_img))).sum()

            # 2. Depth loss within mask
            depth_loss = (torch.abs(rendered_depth - (target_depth))).mean()

            # 3. Silhouette loss to penalize rendering outside mask
            # silhouette_loss = (rendered_silhouette * (1 - object_mask_torch)).sum() * 0.1

            # # 4. Coverage loss to encourage Gaussians to fill the mask
            # coverage_loss = ((1 - rendered_silhouette) * object_mask_torch).sum() * 0.1

            loss = rgb_loss + depth_loss  # + silhouette_loss

            # loss += (torch.abs(rendered_depth - target_depth) * object_mask_torch).mean()
            # Backward pass
            loss.backward(retain_graph=True)
            optimizer.step()

            pbar.set_description(f"Loss: {loss.item():.4f}")

        posquat = torch.tensor(
            video[0].object_poses[object_index], device=device, requires_grad=True
        )
        optimizer = torch.optim.Adam([posquat], lr=5e-4)

        condorgmm.rr_init("splatting_tracking")

        camera_poses = []
        object_poses = []

        pbar = tqdm(range(len(frames)))
        for t in pbar:
            condorgmm.rr_set_time(t)
            mask = video[t].masks[object_index]
            target_img = torch.tensor(
                video[t].rgb * mask[..., None], device=device
            ).float()
            target_depth = torch.tensor(video[t].depth * mask, device=device).float()

            condorgmm.rr_log_rgb(target_img.cpu().detach().numpy(), "image")

            for iteration in range(100):
                optimizer.zero_grad()

                # Forward pass
                rendered_rgb, rendered_depth, rendered_silhouette = render_rgbd(
                    camera_posquat,
                    posquat,
                    means,
                    quats,
                    torch.exp(scales),
                    torch.sigmoid(opacities),
                    rgbs,
                    viewmat[None],
                    K[None],
                    frame0.width,
                    frame0.height,
                )
                if first_frame is None:
                    first_frame = rendered_rgb

                # Compute losses
                # 1. RGB loss within mask
                rgb_loss = (torch.abs(rendered_rgb - (target_img))).sum()

                # 2. Depth loss within mask
                depth_loss = (torch.abs(rendered_depth - (target_depth))).mean()

                # 3. Silhouette loss to penalize rendering outside mask
                # silhouette_loss = (rendered_silhouette * (1 - object_mask_torch)).sum() * 0.1

                # # 4. Coverage loss to encourage Gaussians to fill the mask
                # coverage_loss = ((1 - rendered_silhouette) * object_mask_torch).sum() * 0.1

                loss = rgb_loss + depth_loss  # + silhouette_loss

                # loss += (torch.abs(rendered_depth - target_depth) * object_mask_torch).mean()
                # Backward pass
                loss.backward(retain_graph=True)
                optimizer.step()

            camera_poses.append(camera_posquat.cpu().detach().numpy())
            object_poses.append(posquat.cpu().detach().numpy())

            condorgmm.rr_log_rgb(rendered_rgb.cpu().detach().numpy(), "image/rendered_rgb")

        # get fps from pbar
        fps = num_timesteps / pbar.format_dict["elapsed"]

        from condorgmm import Pose

        gt_poses = [
            Pose(frame.camera_pose).inv() @ Pose(frame.object_poses[object_index])
            for frame in frames
        ]
        predicted_poses = [
            Pose(camera_pose_np).inv() @ Pose(object_pose_np)
            for camera_pose_np, object_pose_np in zip(camera_poses, object_poses)
        ]

        for t in range(len(frames)):
            condorgmm.rr_set_time(t)
            condorgmm.rr_log_pose(predicted_poses[t], "pose")
            condorgmm.rr_log_pose(gt_poses[t], "gt_pose")

        condorgmm.eval.metrics.add_object_tracking_metrics_to_results_dataframe(
            results_df,
            "ycbv",
            "Splatting",
            object_name,
            predicted_poses,
            gt_poses,
            object_mesh.vertices,
            other_info={
                "fps": fps,
                "num_gaussians": means.shape[0],
            },
        )

        aggregated_df = results_df.groupby(
            ["metric", "num_gaussians", "fps", "object", "method"]
        )["value"].apply(lambda x: x.mean())

        print(aggregated_df)

    if experiment_name is None:
        experiment_name = ""
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = f"{experiment_name}_{timestamp}"
    print(f"Experiment name: {experiment_name}")

    results_dir = (
        condorgmm.get_root_path()
        / "results"
        / f"runtime_accuracy_gsplat_{scene_id}_{object_index}_{experiment_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_pickle(results_dir / "runtime_accuracy_results.pkl")


if __name__ == "__main__":
    fire.Fire(run_runtime_accuracy)

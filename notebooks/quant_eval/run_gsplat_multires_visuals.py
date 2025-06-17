import condorgmm
import condorgmm.eval.metrics
import condorgmm.data
from tqdm import tqdm
import condorgmm.camera_tracking
import numpy as np
import condorgmm.warp_gmm.kernels
import fire
import condorgmm
import condorgmm.utils.common.rerun
import datetime as dt
import torch
import condorgmm
import condorgmm.eval.metrics
import condorgmm.data
from condorgmm.ng.torch_utils import render_rgbd
import pickle
import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{os.environ['LD_LIBRARY_PATH']}"
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"


def run_gsplat_multires_visuals(experiment_name=None):
    device = torch.device("cuda:0")

    condorgmm.rr_init("coarse_models_sweep")

    video = condorgmm.data.R3DVideo(
        condorgmm.get_root_path() / "assets/condorgmm_bucket/nearfar.r3d"
    )
    video = video.crop(0, 180, 16, 256)
    frame0 = video[360]

    fx, fy, cx, cy = frame0.intrinsics
    height, width = frame0.depth.shape

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
    2

    object_mask = np.full(frame0.depth.shape, True)

    if experiment_name is None:
        experiment_name = ""
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = f"{experiment_name}_{timestamp}"
    print(f"Experiment name: {experiment_name}")

    results_dir = (
        condorgmm.get_root_path() / "results" / f"gsplat_multires_visuals_{experiment_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    num_gaussians_sweep = [100, 1000, 10000]
    for num_gaussians in num_gaussians_sweep:
        print(f"Running for {num_gaussians} Gaussians")
        sampled_indices = np.random.choice(
            object_mask.sum(), num_gaussians, replace=False
        )

        rgb_means_np = frame0.rgb[object_mask][sampled_indices].astype(np.float32)
        spatial_means_np = condorgmm.utils.common.xyz_from_depth_image(
            frame0.depth, fx, fy, cx, cy
        )[object_mask][sampled_indices].astype(np.float32)

        rgb_means_np = frame0.rgb[object_mask][sampled_indices].astype(np.float32)
        spatial_means_np = condorgmm.utils.common.xyz_from_depth_image(
            frame0.depth, fx, fy, cx, cy
        )[object_mask][sampled_indices].astype(np.float32)
        # Subsample the points

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
            torch.log(torch.randn(num_gaussians, 3) * 0.01),
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
        target_img = torch.tensor(frame0.rgb, device=device).float()
        target_depth = torch.tensor(frame0.depth, device=device).float()

        object_mask_torch = torch.tensor(object_mask, device=device).float()
        # object_mask = torch.tensor(frame0.masks[object_index], device=device).float()

        scales.requires_grad_ = True
        opacities.requires_grad_ = True

        posquat = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=device, requires_grad=True
        )
        camera_posquat = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=device, requires_grad=True
        )

        # Setup optimizer
        optimizer = torch.optim.Adam([means, rgbs, opacities, scales], lr=2e-3)
        n_steps = 1500
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

            # Compute losses
            # 1. RGB loss within mask
            rgb_loss = (
                torch.abs(rendered_rgb - (target_img * object_mask_torch[..., None]))
            ).sum()

            # 2. Depth loss within mask
            depth_loss = (
                torch.abs(rendered_depth - (target_depth * object_mask_torch))
            ).mean()

            # 3. Silhouette loss to penalize rendering outside mask
            silhouette_loss = (
                rendered_silhouette * (1 - object_mask_torch)
            ).sum() * 0.1

            # # 4. Coverage loss to encourage Gaussians to fill the mask
            # coverage_loss = ((1 - rendered_silhouette) * object_mask_torch).sum() * 0.1

            loss = rgb_loss + depth_loss + silhouette_loss

            # loss += (torch.abs(rendered_depth - target_depth) * object_mask_torch).mean()
            # Backward pass
            loss.backward(retain_graph=True)
            optimizer.step()

            pbar.set_description(f"Loss: {loss.item():.4f}")

        results[num_gaussians] = {"image": rendered_rgb.cpu().detach().numpy()}
        results[num_gaussians]["means"] = means.cpu().detach().numpy()
        results[num_gaussians]["quats"] = quats.cpu().detach().numpy()
        results[num_gaussians]["log_scales"] = scales.cpu().detach().numpy()
        results[num_gaussians]["log_opacities"] = opacities.cpu().detach().numpy()
        results[num_gaussians]["rgb"] = rgbs.cpu().detach().numpy()


        # Save results dictionary to pickle file
        results_path = results_dir / "gsplat_multires_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    fire.Fire(run_gsplat_multires_visuals)

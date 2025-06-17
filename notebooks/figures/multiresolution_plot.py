### The code for this matplotlib layout was
### generated in this ChatGPT conversation:
### https://chatgpt.com/share/67c7972e-0bfc-8009-bfd1-6140dca2f1bc.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import condorgmm
from condorgmm.utils.common import get_assets_path
import condorgmm.data as data
from condorgmm.condor.interface.camera_tracking import initialize, fast_config
from condorgmm.condor.rerun import log_state
from condorgmm.condor.pose import Pose
import jax.numpy as jnp
import jax
from condorgmm.data.ycb_dataloader import YCB_MODEL_NAMES
from condorgmm.condor.interface.shared import _frame_to_intrinsics
from condorgmm.condor.interface.object_tracking import fit_object_model, default_cfg
from condorgmm.condor.types import Gaussian
from condorgmm.condor.geometry import isovars_and_quaternion_to_cov
from jax.random import key as prngkey
from condorgmm.condor.rerun import _log_gaussians
from condorgmm.condor.pose import Pose as CondorPose
import rerun as rr
import trimesh
import glob
import pandas as pd
import condorgmm.eval.fp_loader
import pickle


def gaussian_count_str(n):
    if n < 1000:
        return f"{n} Gaussians"
    return f"{n//1000}K Gaussians"


condorgmm_results_directory = (
    condorgmm.get_root_path()
    / "assets/condorgmm_bucket/runtime_accuracy_3_2__2025-03-20-19-55-42"
)

splatting_results_directory = (
    condorgmm.get_root_path()
    / "assets/condorgmm_bucket/runtime_accuracy_gsplat_3_2__2025-03-20-19-53-16"
)


def make_top_row(fig, gs_outer_top, top_left_images, top_right_images, gaussian_counts):
    """
    Creates the top row of the figure:
      - Left group: a narrow column for row labels ("Gaussians" and "Reconstruction") and a 3×2 grid of images.
      - Right group: a 3×2 grid of images.
    Adds column titles to both 3×2 grids and places overall labels ("condorgmm" and "Gaussian Splatting")
    above the groups.

    Parameters:
      fig: the matplotlib Figure.
      gs_outer_top: a GridSpec for the top row.
      top_left_images: a 2×3 array (list of lists) of images for the left grid.
      top_right_images: a 2×3 array of images for the right grid.
      gaussian_counts: a list of three integers - number of Gaussians for each of the 3 cols.

    Returns:
      axes_left, axes_right: nested lists of Axes from the left and right 3×2 image grids.
    """
    # Split the top row into two groups: left and right.
    gs_top_outer = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer_top, width_ratios=[0.6, 0.4], wspace=0.05
    )

    # --- Left Group: Nested grid for row labels and left 3×2 images ---
    gs_left_group = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_top_outer[0], width_ratios=[0.06, 0.94], wspace=0.0
    )

    # Row Labels (left column)
    gs_row_labels = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_left_group[0], hspace=0.0
    )
    ax_label_top = fig.add_subplot(gs_row_labels[0])
    ax_label_bot = fig.add_subplot(gs_row_labels[1])
    ax_label_top.axis("off")
    ax_label_bot.axis("off")
    ax_label_top.text(
        1.0,
        0.5,
        "Gaussians",
        ha="right",
        va="center",
        fontsize=10,
        transform=ax_label_top.transAxes,
    )
    ax_label_bot.text(
        1.0,
        0.5,
        "Reconstruction",
        ha="right",
        va="center",
        fontsize=10,
        transform=ax_label_bot.transAxes,
    )

    # Left 3×2 grid of images
    gs_left_images = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs_left_group[1], wspace=0.0, hspace=0.1
    )
    axes_left = [
        [fig.add_subplot(gs_left_images[i, j]) for j in range(3)] for i in range(2)
    ]

    # Populate left grid using provided images.
    for i in range(2):
        for j in range(3):
            axes_left[i][j].imshow(top_left_images[i][j], cmap="gray")
            axes_left[i][j].axis("off")

    # --- Right Group: Right 3×2 grid of images ---
    gs_right_images = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs_top_outer[1], wspace=0.0, hspace=0.1
    )
    axes_right = [
        [fig.add_subplot(gs_right_images[i, j]) for j in range(3)] for i in range(2)
    ]

    # Populate right grid using provided images.
    for i in range(2):
        for j in range(3):
            axes_right[i][j].imshow(top_right_images[i][j], cmap="gray")
            axes_right[i][j].axis("off")

    # Add column titles above each grid.

    col_titles = [gaussian_count_str(n) for n in gaussian_counts]
    for j, title in enumerate(col_titles):
        axes_left[0][j].set_title(title, fontsize=10)
        axes_right[0][j].set_title(title, fontsize=10)

    plt.draw()  # Update layout positions.

    # Compute centers for overall labels.
    pos_left_first = axes_left[0][0].get_position()
    pos_left_last = axes_left[0][-1].get_position()
    center_left = (pos_left_first.x0 + pos_left_last.x1) / 2

    pos_right_first = axes_right[0][0].get_position()
    pos_right_last = axes_right[0][-1].get_position()
    center_right = (pos_right_first.x0 + pos_right_last.x1) / 2

    # Place overall labels above the image grids.
    fig.text(center_left, 0.9, "condorgmm", ha="center", fontsize=12)
    fig.text(center_right, 0.9, "Gaussian Splatting", ha="center", fontsize=12)

    return axes_left, axes_right


def make_bottom_left_grid(fig, gs_bottom_left, bottom_left_images, gaussian_counts):
    """
    Creates the bottom left 3×2 grid of images with column labels and a group title.
    Column labels: "20 Gaussians", "200 Gaussians", "2K Gaussians"
    Group title: "condorgmm Representations of Objects"

    Parameters:
      fig: the matplotlib Figure.
      gs_bottom_left: the GridSpec for the bottom left group.
      bottom_left_images: a 2×3 array of images for this grid.

    Returns:
      axes: a 2×3 list of Axes.
    """
    gs_bottom_grid = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs_bottom_left, wspace=0.2, hspace=0.1
    )
    axes = [[fig.add_subplot(gs_bottom_grid[i, j]) for j in range(3)] for i in range(2)]

    # Set column labels on the top row.
    col_labels = [gaussian_count_str(n) for n in gaussian_counts]
    for j, label in enumerate(col_labels):
        axes[0][j].set_title(label, fontsize=10)

    # Populate each axis with the provided image.
    for i in range(2):
        for j in range(3):
            axes[i][j].imshow(bottom_left_images[i][j], cmap="gray")
            axes[i][j].axis("off")

    plt.draw()  # Update layout.

    # Move all axes up by 0.03
    for i in range(2):
        for j in range(3):
            pos = axes[i][j].get_position()
            new_pos = [pos.x0, pos.y0 + 0.03, pos.width, pos.height]
            axes[i][j].set_position(new_pos)

    # Compute the center of the top row for title placement.
    pos_first = axes[0][0].get_position()
    pos_last = axes[0][-1].get_position()
    center_x = (pos_first.x0 + pos_last.x1) / 2
    top_y = axes[0][0].get_position().y1
    fig.text(
        center_x,
        top_y + 0.04,
        "condorgmm Representations of Objects",
        ha="center",
        fontsize=12,
    )

    return axes


def get_plot_data():
    all_results_dfs = []
    method_names = []

    results_files = glob.glob(str(condorgmm_results_directory) + "/*.pkl")
    results_dfs = [pd.read_pickle(f) for f in results_files]
    assert len(results_dfs) == 1
    results_df = results_dfs[0]
    all_results_dfs.append(results_df)
    method_names.append("condorgmm")

    assert splatting_results_directory is not None
    splatting_results_files = glob.glob(str(splatting_results_directory) + "/*.pkl")
    splatting_results_dfs = [pd.read_pickle(f) for f in splatting_results_files]
    assert len(splatting_results_dfs) == 1
    splatting_results_df = splatting_results_dfs[0]
    all_results_dfs.append(splatting_results_df)
    method_names.append("GSplat")

    values = {}
    min_n_gaussians = 10000000000
    max_n_gaussians = 0

    for results_df, method_name in zip(all_results_dfs, method_names):
        aggregated_df = results_df.groupby(["metric", "num_gaussians", "fps"])[
            "value"
        ].apply(lambda x: x.mean())

        num_gaussians_list = []
        runtime_list = []
        add_list = []

        for index, value in aggregated_df["ADD"].items():
            num_gaussians, fps = index
            num_gaussians_list.append(num_gaussians)
            runtime_list.append(fps)
            add_list.append(value * 100.0)
            min_n_gaussians = min(min_n_gaussians, num_gaussians)
            max_n_gaussians = max(max_n_gaussians, num_gaussians)

        values[method_name] = {
            "num_gaussians_list": num_gaussians_list,
            "runtime_list": runtime_list,
            "add_list": add_list,
        }

    # Get FP ADD
    fp_ycbv_result = condorgmm.eval.fp_loader.YCBVTrackingResultLoader(
        frame_rate=1, split="train_real"
    )
    df = fp_ycbv_result.get_dataframe(all_results_dfs[0]["scene"].iloc[0])
    df = df[df["timestep"] < 100]
    df = df[df["object"] == results_df["object"].iloc[0]]
    aggregated_df = df.groupby(["metric"])["value"].apply(lambda x: x.mean())
    fp_add = aggregated_df["ADD"] * 100.0
    num_gaussians_list = [min_n_gaussians, max_n_gaussians]
    values["FP"] = {
        "num_gaussians_list": [min_n_gaussians, max_n_gaussians],
        "add_list": [fp_add for _ in range(len(num_gaussians_list))],
        "runtime_list": [30.0 for _ in range(len(num_gaussians_list))],
    }

    return values


def make_line_plots(fig, gs_plot_left, gs_plot_right, line_plot_data):
    """
    Creates two line plots, each showing three lines with labels:
      "condorgmm", "Gaussian Splatting", and "FoundationPose".
    Adds an overall title "Gaussian Count vs Performance & Accuracy" above the plots,
    and a shared legend to the right of the plots.
    Moves the entire line plots group a bit higher.

    Parameters:
      fig: the matplotlib Figure.
      gs_plot_left: GridSpec for the first plot.
      gs_plot_right: GridSpec for the second plot.
      line_plot_data: a dict with keys "plot1" and "plot2". Each maps to a dict
                      whose keys are the labels and values are tuples (x, y) data.

    Returns:
      The two Axes for the line plots.
    """
    colors = {
        "condorgmm": "black",
        "Gaussian Splatting": "red",
        "FoundationPose": "blue",
    }

    ax_plot1 = fig.add_subplot(gs_plot_left)
    ax_plot2 = fig.add_subplot(gs_plot_right)

    # Set log scale for x-axis, with ticks at each power of 10.
    ax_plot1.set_xscale("log", base=10)
    ax_plot2.set_xscale("log", base=10)
    ax_plot1.xaxis.set_major_locator(plt.LogLocator(base=10))
    ax_plot2.xaxis.set_major_locator(plt.LogLocator(base=10))

    # Plot data for the first plot.
    for label, (x, y) in line_plot_data["plot1"].items():
        ax_plot1.plot(x, y, label=label, color=colors[label])

    # Plot data for the second plot.
    for label, (x, y) in line_plot_data["plot2"].items():
        ax_plot2.plot(x, y, label=label, color=colors[label])

    # Set axis labels.
    ax_plot1.set_xlabel("Number of Gaussians")
    ax_plot1.set_ylabel("Frames per second")
    ax_plot2.set_xlabel("Number of Gaussians")
    ax_plot2.set_ylabel("Mean ADD (cm)")

    plt.draw()

    # Shift the entire group upward.
    delta = 0.05  # shift upward by 0.05 in figure coordinates
    for ax in [ax_plot1, ax_plot2]:
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0 + delta, pos.width, pos.height]
        ax.set_position(new_pos)

    # Recompute overall title position from the new axes positions.
    pos_left = ax_plot1.get_position()
    pos_right = ax_plot2.get_position()
    center_x = (pos_left.x0 + pos_right.x1) / 2
    top_y = max(pos_left.y1, pos_right.y1)
    fig.text(
        center_x,
        top_y + 0.02,
        "Number of Gaussians vs Speed & Accuracy",
        ha="center",
        fontsize=12,
    )

    # Create a shared legend using handles from the first plot.
    handles = ax_plot1.get_lines()
    labels = [h.get_label() for h in handles]
    fig.legend(
        handles, labels, loc="center right", bbox_to_anchor=(0.9, 0.35), fontsize=10
    )

    return ax_plot1, ax_plot2


def create_figure(
    top_left_images,
    top_right_images,
    toprow_gaussian_counts,
    bottom_left_images,
    bottom_left_gaussian_counts,
    line_plot_data,
):
    """
    Creates the complete figure by assembling the top row (two 3×2 grids of images),
    the bottom left (a 3×2 grid of images), and the bottom right (two line plots).

    Parameters:
      top_left_images: 2×3 array of images for the top-left grid.
      top_right_images: 2×3 array of images for the top-right grid.
      bottom_left_images: 2×3 array of images for the bottom-left grid.
      line_plot_data: dict with keys "plot1" and "plot2", each mapping to a dict of {label: (x, y)}.

    Returns:
      The assembled matplotlib Figure.
    """
    fig = plt.figure(figsize=(12, 8))
    gs_outer = gridspec.GridSpec(2, 1, height_ratios=[0.6, 0.4], hspace=0.4, figure=fig)

    # Top row.
    axes_left, axes_right = make_top_row(
        fig, gs_outer[0], top_left_images, top_right_images, toprow_gaussian_counts
    )

    # Bottom row: Create a 1×3 layout.
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_outer[1], width_ratios=[0.7, 0.45, 0.45], wspace=0.4
    )
    # Bottom left grid.
    make_bottom_left_grid(
        fig, gs_bottom[0], bottom_left_images, bottom_left_gaussian_counts
    )
    # Bottom right: two line plots.
    make_line_plots(fig, gs_bottom[1], gs_bottom[2], line_plot_data)

    return fig


def fit_object_models_and_log_to_rerun(
    ycb_object_name,  # string; an element of YCB_MODEL_NAMES
    gaussian_counts,  # list of natural numbers
):
    original_video = data.YCBVVideo.training_scene(0)
    intrinsics = _frame_to_intrinsics(original_video[0])
    mesh: trimesh.Trimesh = original_video.get_object_mesh_from_id(
        YCB_MODEL_NAMES.index(ycb_object_name)
    )

    pairwise_vertex_dists = np.linalg.norm(
        mesh.vertices[:, None] - mesh.vertices[None], axis=-1
    )
    max_dist = np.max(pairwise_vertex_dists)

    # Get mesh vertices in a form suitable for rendering
    xyz = np.array(mesh.vertices)
    rgb = np.zeros((len(xyz), 3), dtype=np.uint8)
    rgb[:] = [200, 200, 200]  # Default light grey color for visualization

    condorgmm.rr_init(f"object-fitting--{ycb_object_name}-00")
    for i, cnt in enumerate(gaussian_counts):
        print(f"Fitting {cnt} gaussians to the {ycb_object_name}...")
        cfg = default_cfg.replace(
            n_gaussians_for_object=cnt, n_pts_for_object_fitting=100000
        )
        hyp = cfg.base_hypers.replace(n_gaussians=cnt + 1, intrinsics=intrinsics)

        object_model, _ = fit_object_model(
            prngkey(1), mesh, hyp, cfg, k_for_initialization=25
        )

        rr.set_time_sequence("fit", i)

        covs = object_model.xyz_cov
        max_evals = jax.vmap(lambda cov: jnp.linalg.eigh(cov)[0][-1])(covs)
        object_model = object_model[max_evals < max_dist / cnt]

        _log_gaussians(
            object_model.replace(
                xyz_cov=object_model.xyz_cov * 2**2,
            ),
            object_model.mixture_weight / object_model.mixture_weight.sum()
            > 10 / len(object_model),
            CondorPose.identity(),
            fill_mode=rr.components.FillMode.MajorWireframe,
        )
        rr.log("n_gaussians", rr.TextDocument(f"n gaussians = {cnt}"))
        print("...done!")


def generate_rerun_logs_for_object_panels():
    """
    Run this to generate the visualizations that were screenshotted
    for the bottom left panel.
    """
    cnts = [20, 200, 2000]
    fit_object_models_and_log_to_rerun("011_banana", cnts)
    fit_object_models_and_log_to_rerun("003_cracker_box", cnts)


def generate_multires_scenes_and_log_to_rerun():
    """
    Run this to generate the visualizations that were screenshotted
    for the top left panel.
    """
    video = data.R3DVideo(get_assets_path() / "nearfar.r3d")
    video = video.crop(0, 180, 16, 256)
    frame = video[360]

    cfg100 = fast_config.replace(
        n_gaussians=100,
        tile_size_x=32,
        tile_size_y=32,
        repopulate_depth_nonreturns=False,
    )
    cfg1000 = fast_config.replace(
        n_gaussians=1000,
        tile_size_x=8,
        tile_size_y=8,
        repopulate_depth_nonreturns=False,
    )
    cfg10000 = fast_config.replace(
        n_gaussians=10000,
        tile_size_x=2,
        tile_size_y=2,
        repopulate_depth_nonreturns=False,
    )

    condorgmm.rr_init("multires-scene-fitting-00")
    Solid = rr.components.FillMode.Solid

    _, ccts100 = initialize(frame, condorgmm.Pose.identity(), cfg100)
    rr.set_time_sequence("fit", 0)
    log_state(ccts100.state, ccts100.hypers, ellipse_mode=Solid, ellipse_scalar=2)
    rr.log("n_gaussians", rr.TextDocument("n gaussians = 100"))

    _, ccts1000 = initialize(frame, condorgmm.Pose.identity(), cfg1000)
    rr.set_time_sequence("fit", 1)
    log_state(ccts1000.state, ccts1000.hypers, ellipse_mode=Solid, ellipse_scalar=2)
    rr.log("n_gaussians", rr.TextDocument("n gaussians = 1000"))

    _, ccts10000 = initialize(frame, condorgmm.Pose.identity(), cfg10000)
    rr.set_time_sequence("fit", 2)
    log_state(ccts10000.state, ccts10000.hypers, ellipse_mode=Solid, ellipse_scalar=2)
    rr.log("n_gaussians", rr.TextDocument("n gaussians = 10000"))


def _splat_dict_to_gaussian(d):
    n_gaussians = d["means"].shape[0]
    log_scales = jnp.array(d["log_scales"])

    is_valid = jax.vmap(lambda x: jnp.logical_not(jnp.any(jnp.isnan(x))))(log_scales)
    # mx = jnp.max(jnp.array(d['log_opacities']))
    # is_valid = jnp.logical_and(is_valid, jnp.array(d['log_opacities']) > mx - 3.0)

    log_scales = jnp.nan_to_num(log_scales, nan=-10.0)
    xyz_cov = jax.vmap(isovars_and_quaternion_to_cov)(
        jnp.exp(log_scales), jnp.array(d["quats"])
    )
    return Gaussian(
        idx=jnp.arange(n_gaussians),
        xyz=jnp.array(d["means"]),
        xyz_cov=xyz_cov,
        rgb=jnp.array(d["rgb"]),
        rgb_vars=jnp.ones_like(d["rgb"]),
        mixture_weight=jnp.ones(n_gaussians),
        origin=jnp.ones(n_gaussians),
        object_idx=jnp.zeros(n_gaussians, dtype=int),
        n_frames_since_last_had_assoc=jnp.zeros(n_gaussians),
    )[is_valid]


def _log_gaussian_splatting_results_to_rerun(
    idx, n_gaussians, gaussians, ellipse_scalar=2
):
    rr.set_time_sequence("fit", idx)
    gaussians = gaussians.replace(xyz_cov=gaussians.xyz_cov * ellipse_scalar**2)
    _log_gaussians(
        gaussians,
        jnp.ones(len(gaussians)),
        Pose.identity(),
        fill_mode=rr.components.FillMode.Solid,
    )
    rr.log("n_gaussians", rr.TextDocument(f"n gaussians = {n_gaussians}"))


def log_gaussian_splatting_results_to_rerun():
    condorgmm.rr_init("gaussian-splatting-multires-00")
    gspal_multires_visuals_directory = (
        condorgmm.get_root_path()
        / "assets/condorgmm_bucket/gsplat_multires_visuals__2025-04-08-18-28-53"
    )
    results_files = glob.glob(str(gspal_multires_visuals_directory) + "/*.pkl")
    data = pickle.load(open(results_files[0], "rb"))
    num_gaussians_list = list(data.keys())
    num_gaussians_list.sort()
    print(data[num_gaussians_list[0]].keys())

    for idx, n_gaussians in enumerate(num_gaussians_list):
        gaussians = _splat_dict_to_gaussian(data[n_gaussians])
        _log_gaussian_splatting_results_to_rerun(idx, n_gaussians, gaussians)


# Example usage with dummy data:
if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'

    data = get_plot_data()
    line_plot_data = {
        "plot1": {
            "condorgmm": (
                data["condorgmm"]["num_gaussians_list"],
                data["condorgmm"]["runtime_list"],
            ),
            "Gaussian Splatting": (
                data["GSplat"]["num_gaussians_list"],
                data["GSplat"]["runtime_list"],
            ),
            "FoundationPose": (
                data["FP"]["num_gaussians_list"],
                data["FP"]["runtime_list"],
            ),
        },
        "plot2": {
            "condorgmm": (data["condorgmm"]["num_gaussians_list"], data["condorgmm"]["add_list"]),
            "Gaussian Splatting": (
                data["GSplat"]["num_gaussians_list"],
                data["GSplat"]["add_list"],
            ),
            "FoundationPose": (
                data["FP"]["num_gaussians_list"],
                data["FP"]["add_list"],
            ),
        },
    }

    colors = {
        "condorgmm": "black",
        "Gaussian Splatting": "red",
        "FoundationPose": "blue",
    }
    line_styles = {
        "condorgmm": "-",
        "Gaussian Splatting": "-",
        "FoundationPose": "--",
    }
    fig = plt.figure(figsize=(5, 10))
    plt.subplot(2, 1, 2)
    
    fontsize = 20

    # Set log scale for x-axis, with ticks at each power of 10.
    plt.xscale("log", base=10)
    plt.gca().xaxis.set_major_locator(plt.LogLocator(base=10))

    # Plot data for the first plot.
    for label, (x, y) in line_plot_data["plot1"].items():
        plt.plot(x, y, label=label, color=colors[label], linestyle=line_styles[label], linewidth=3)
        
    plt.title("Speed vs Number of Gaussians", fontsize=fontsize)

    # Set axis labels.
    plt.xlabel("Number of Gaussians", fontsize=fontsize-3)
    plt.ylabel("Frames per second", fontsize=fontsize-3)
    
    # Make plot square
    ax = plt.gca()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    
    plt.subplot(2, 1, 1)

    # Plot data for the second plot.
    for label, (x, y) in line_plot_data["plot2"].items():
        plt.plot(x, y, label=label, color=colors[label], linestyle=line_styles[label], linewidth=3)

    plt.xscale("log", base=10)
    plt.gca().xaxis.set_major_locator(plt.LogLocator(base=10))

    plt.title("Accuracy vs Number of Gaussians", fontsize=fontsize)

    plt.xlabel("Number of Gaussians", fontsize=fontsize-3)
    plt.ylabel("Mean ADD Error (cm)", fontsize=fontsize-3)
    
    # Make plot square
    ax = plt.gca()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    
    fig.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(0.5, -0.0),
        loc="center",
        ncol=1,
        fontsize=fontsize-3,
    )

    plt.subplots_adjust(hspace=0.3)

    plt.savefig("multires_plot.pdf", bbox_inches="tight")

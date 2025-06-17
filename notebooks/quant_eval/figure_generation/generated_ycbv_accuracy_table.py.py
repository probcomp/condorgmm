import pandas as pd
import glob
import condorgmm.eval.metrics
import condorgmm.data
import numpy as np
import fire
import condorgmm
import condorgmm.eval.fp_loader
import condorgmm
import matplotlib.pyplot as plt

results_directory = (
    condorgmm.get_root_path()
    / "assets/condorgmm_bucket/runtime_accuracy_3_2__2025-03-20-19-55-42"
)


def ycbv_pose_tracking_table(results_directory=None):
    if results_directory is None:
        results_directory = (
            condorgmm.get_root_path()
            / "assets/condorgmm_bucket/object_pose_tracking_ycbv_test_em__2025-03-05-02-11-29"
        )
    results_files = glob.glob(str(results_directory) + "/*.pkl")
    results_dfs = [pd.read_pickle(f) for f in results_files]
    concatenated_df = pd.concat(results_dfs)

    video = condorgmm.data.YCBTestVideo(48)

    results_df = results_dfs[0]
    scene, object_name = results_df.iloc[0]["scene"], results_df.iloc[0]["object"]
    max_T = results_df["timestep"].max()

    ## Load FoundationPose results
    ycb_dir = video.ycb_dir

    all_foundation_pose_results_df = []
    fp_loader = condorgmm.eval.fp_loader.YCBVTrackingResultLoader(
        frame_rate=1, split=ycb_dir.name
    )

    for scene in video.SCENE_NAMES:
        print("Scene: ", scene)
        video = condorgmm.data.YCBTestVideo(scene)

        df = fp_loader.get_dataframe(scene)
        df = df[df["timestep"] < max_T + 1]
        all_foundation_pose_results_df.append(df)

    all_foundation_pose_results_df_concatenated = pd.concat(
        all_foundation_pose_results_df
    )

    all_se3_results_loaded = pd.read_pickle(
        condorgmm.get_root_path() / "assets" / "se3_results" / "se3_results_df_400.pkl"
    )

    assert (
        len(all_foundation_pose_results_df_concatenated)
        == len(concatenated_df)
        == len(all_se3_results_loaded)
    )

    # Aggregate all results
    all_results_df = pd.concat(
        [
            concatenated_df,
            all_foundation_pose_results_df_concatenated,
            all_se3_results_loaded,
        ]
    )

    auc_results = all_results_df.groupby(["metric", "object", "method"])["value"].apply(
        condorgmm.eval.metrics.compute_auc
    )
    print(auc_results)

    mean_results = all_results_df.groupby(["metric", "method"])["value"].apply(np.mean)
    print(mean_results)

    std_results = all_results_df.groupby(["metric", "method"])["value"].apply(np.std)
    print(std_results)

    model_names = condorgmm.data.YCBTestVideo.YCB_MODEL_NAMES
    methods = ["SE3-Tracknet", "FoundationPose", "condorgmm"]

    table_string = """"""
    for object_name in model_names:
        add_results = [
            auc_results["ADD"][object_name][method] * 100.0 for method in methods
        ]
        adds_results = [
            auc_results["ADD-S"][object_name][method] * 100.0 for method in methods
        ]

        # Get indices sorted by score (descending)
        add_sorted_indices = np.argsort(add_results)[::-1]
        adds_sorted_indices = np.argsort(adds_results)[::-1]

        object_name_mod = object_name.replace(r"_", r"\_")
        table_string += rf"{object_name_mod}"

        for i in range(len(methods)):
            # Color coding for ADD
            if i == add_sorted_indices[0]:
                table_string += (
                    r" & \cellcolor{green}{"
                    + rf"\textbf{{{add_results[i]:.1f}}}"
                    + r"}"
                )
            elif i == add_sorted_indices[1]:
                table_string += (
                    r" & \cellcolor{yellow}{" + rf"{add_results[i]:.1f}" + r"}"
                )
            else:
                table_string += (
                    r" & \cellcolor{orange}{" + rf"{add_results[i]:.1f}" + r"}"
                )

            # Color coding for ADD-S
            if i == adds_sorted_indices[0]:
                table_string += (
                    r" & \cellcolor{green}{"
                    + rf"\textbf{{{adds_results[i]:.1f}}}"
                    + r"}"
                )
            elif i == adds_sorted_indices[1]:
                table_string += (
                    r" & \cellcolor{yellow}{" + rf"{adds_results[i]:.1f}" + r"}"
                )
            else:
                table_string += (
                    r" & \cellcolor{orange}{" + rf"{adds_results[i]:.1f}" + r"}"
                )

        table_string += r"\\" + "\n"

    table_string += r"\hline" + "\n"

    auc_results_full = all_results_df.groupby(["metric", "method"])["value"].apply(
        condorgmm.eval.metrics.compute_auc
    )
    print(auc_results_full)

    add_results = [auc_results_full["ADD"][method] * 100.0 for method in methods]
    adds_results = [auc_results_full["ADD-S"][method] * 100.0 for method in methods]

    # Get indices sorted by score (descending)
    add_sorted_indices = np.argsort(add_results)[::-1]
    adds_sorted_indices = np.argsort(adds_results)[::-1]

    table_string += r"All Frames"

    for i in range(len(methods)):
        # Color coding for ADD
        if i == add_sorted_indices[0]:
            table_string += (
                r" & \cellcolor{green}{" + rf"\textbf{{{add_results[i]:.1f}}}" + r"}"
            )
        elif i == add_sorted_indices[1]:
            table_string += r" & \cellcolor{yellow}{" + rf"{add_results[i]:.1f}" + r"}"
        else:
            table_string += r" & \cellcolor{orange}{" + rf"{add_results[i]:.1f}" + r"}"

        # Color coding for ADD-S
        if i == adds_sorted_indices[0]:
            table_string += (
                r" & \cellcolor{green}{" + rf"\textbf{{{adds_results[i]:.1f}}}" + r"}"
            )
        elif i == adds_sorted_indices[1]:
            table_string += r" & \cellcolor{yellow}{" + rf"{adds_results[i]:.1f}" + r"}"
        else:
            table_string += r" & \cellcolor{orange}{" + rf"{adds_results[i]:.1f}" + r"}"

    table_string += r"\\" + "\n"

    full_table_string = (
        r"""
    \begin{tabular}{lcccccc}
    \toprule
    \textbf{Method}& \multicolumn{2}{c}{SE3-Tracknet} &  \multicolumn{2}{c}{FoundationPose} & \multicolumn{2}{c}{condorgmm} \\
    \midrule
    \textbf{Object} & ADD & ADD-S & ADD & ADD-S & ADD & ADD-S \\
    \midrule
    """
        + table_string
        + r"\midrule FPS & \multicolumn{2}{c}{90.1} & \multicolumn{2}{c}{30.0} & \multicolumn{2}{c}{40.1} \\ \bottomrule \end{tabular}"
    )

    tex_dir = condorgmm.get_root_path() / "condorgmm_tex"
    with open(tex_dir / "ycbv_pose_tracking_table.tex", "w") as f:
        f.write(full_table_string)

    # Filter for ADD metric only
    add_df = all_results_df[all_results_df["metric"] == "ADD"]
    adds_df = all_results_df[all_results_df["metric"] == "ADD-S"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

    # Define fixed bins for both plots
    max_val = 0.03
    num_bins = 100
    bins = np.linspace(0, max_val, num_bins)  # 50 bins between 0 and 0.03
    fontsize = 22

    se3_tracknet_color = "orange"
    foundation_pose_color = "gold"
    condorgmm_color = "green"

    # Plot ADD histogram
    ax1.hist(
        add_df[add_df["method"] == "SE3-Tracknet"]["value"],
        bins=bins,
        alpha=0.5,
        label="SE3-Tracknet",
        color=se3_tracknet_color,
    )
    ax1.hist(
        add_df[add_df["method"] == "FoundationPose"]["value"],
        bins=bins,
        alpha=0.5,
        label="FoundationPose",
        color=foundation_pose_color,
    )
    ax1.hist(
        add_df[add_df["method"] == "condorgmm"]["value"],
        bins=bins,
        alpha=0.5,
        label="condorgmm",
        color=condorgmm_color,
    )
    ax1.set_xlabel("ADD Value", fontsize=fontsize)
    ax1.set_ylabel("Count", fontsize=fontsize)
    ax1.set_xlim(0, 0.03)
    ax1.set_title("ADD", fontsize=fontsize + 5)

    # Plot ADD-S histogram
    max_val = 0.015
    bins = np.linspace(0, max_val, num_bins)  # 50 bins between 0 and 0.03
    ax2.hist(
        adds_df[adds_df["method"] == "SE3-Tracknet"]["value"],
        bins=bins,
        alpha=0.5,
        label="SE3-Tracknet",
        color=se3_tracknet_color,
    )
    ax2.hist(
        adds_df[adds_df["method"] == "FoundationPose"]["value"],
        bins=bins,
        alpha=0.5,
        label="FoundationPose",
        color=foundation_pose_color,
    )
    ax2.hist(
        adds_df[adds_df["method"] == "condorgmm"]["value"],
        bins=bins,
        alpha=0.5,
        label="condorgmm",
        color=condorgmm_color,
    )
    ax2.set_xlabel("ADD-S Value", fontsize=fontsize)
    ax2.set_ylabel("Count", fontsize=fontsize)
    ax2.set_xlim(0, 0.015)
    ax2.set_title("ADD-S", fontsize=fontsize + 5)

    # Place legend below both subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(0.5, -0.1),
        loc="center",
        ncol=3,
        fontsize=fontsize - 3,
    )

    plt.subplots_adjust(hspace=0.5)

    plt.tight_layout()

    tex_dir = condorgmm.get_root_path() / "condorgmm_tex"
    plt.savefig(tex_dir / "add_adds_histogram.pdf", bbox_inches="tight")


if __name__ == "__main__":
    fire.Fire(ycbv_pose_tracking_table)

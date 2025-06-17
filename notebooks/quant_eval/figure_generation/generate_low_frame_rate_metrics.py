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

import condorgmm.eval.fp_loader

def ycbv_pose_tracking_table(results_directory=None):
    if results_directory is None:
        results_directory = (
            condorgmm.get_root_path()
            / "results/object_pose_tracking_ycbv_test_low_frame_rate__2025-04-14-03-55-37"
        )
    results_files = glob.glob(str(results_directory) + "/*.pkl")
    results_dfs = [pd.read_pickle(f) for f in results_files]
    concatenated_df = pd.concat(results_dfs)

    video = condorgmm.data.YCBTestVideo(48)
    ycb_dir = video.ycb_dir
    
    fp_loader = condorgmm.eval.fp_loader.YCBVTrackingResultLoader(
        frame_rate=50, split=ycb_dir.name
    )
    fp_dfs = []
    for scene in video.SCENE_NAMES:
        fp_dfs.append(fp_loader.get_dataframe(scene))
    
    fp_df = pd.concat(fp_dfs)
    
    
    assert len(fp_df) == len(concatenated_df)

    all_results_df = pd.concat([fp_df, concatenated_df])
    auc_results = all_results_df.groupby(["metric", "object", "method", ])["value"].apply(
        condorgmm.eval.metrics.compute_auc
    )
    pd.set_option('display.max_rows', None) 
    pd.set_option('display.max_columns', None)
    print(auc_results)
    


    
    # Get values for each object
    fp_values_add = []
    condorgmm_values_add = []
    fp_values_adds = []
    condorgmm_values_adds = []
    object_names = []
    
    model_names = condorgmm.data.YCBTestVideo.YCB_MODEL_NAMES
    for obj in model_names:
        # ADD values
        fp_val = auc_results["ADD"][obj]["FoundationPose"] * 100.0
        condorgmm_val = auc_results["ADD"][obj]["condorgmm"] * 100.0
        fp_values_add.append(fp_val)
        condorgmm_values_add.append(condorgmm_val)
        
        # ADD-S values
        fp_val_s = auc_results["ADD-S"][obj]["FoundationPose"] * 100.0
        condorgmm_val_s = auc_results["ADD-S"][obj]["condorgmm"] * 100.0
        fp_values_adds.append(fp_val_s)
        condorgmm_values_adds.append(condorgmm_val_s)
        
        object_names.append(obj)
    # Create scatter plots comparing FoundationPose vs condorgmm performance
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fontsize = 30
    
    # Plot ADD points
    # Add background coloring
    min_val = 0.0
    max_val = 100.0
    
    ax1.scatter(fp_values_add, condorgmm_values_add, s=100)
    ax1.set_xlabel('FoundationPose ADD AUC', fontsize=fontsize)
    ax1.set_ylabel('condorgmm ADD AUC', fontsize=fontsize)
    ax1.set_title('ADD Metric Comparison', fontsize=fontsize+10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax1.set_aspect('equal', adjustable='box')
    # Add diagonal line
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    # Add green shading above diagonal line
    x = np.linspace(min_val, max_val, 100)
    y = np.linspace(min_val, max_val, 100)
    X, Y = np.meshgrid(x, y)
    ax1.fill_between(x, x, max_val, alpha=0.1, color='green')
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)

    # Plot ADD-S points
    # Add background coloring
    
    ax2.scatter(fp_values_adds, condorgmm_values_adds, s=100)
    ax2.set_xlabel('FoundationPose ADD-S AUC', fontsize=fontsize)
    ax2.set_ylabel('condorgmm ADD-S AUC', fontsize=fontsize)

    ax2.set_title('ADD-S Metric Comparison', fontsize=fontsize+10)
    # Set equal aspect ratio after setting limits
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    ax2.fill_between(x, x, max_val, alpha=0.1, color='green')
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)

    plt.tight_layout()
    tex_dir = condorgmm.get_root_path() / "condorgmm_tex"
    plt.savefig(tex_dir / "add_auc_scatter.pdf")
    model_names = condorgmm.data.YCBTestVideo.YCB_MODEL_NAMES
    methods = ["FoundationPose", "condorgmm"]

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
    \begin{tabular}{lcccc}
    \toprule
    \textbf{Method}& \multicolumn{2}{c}{FoundationPose} & \multicolumn{2}{c}{condorgmm} \\
    \midrule
    \textbf{Object} & ADD & ADD-S & ADD & ADD-S \\
    \midrule
    """
        + table_string
        + r"\bottomrule \end{tabular}"
    )

    tex_dir = condorgmm.get_root_path() / "condorgmm_tex"
    with open(tex_dir / "ycbv_low_frame_rate_pose_tracking_table.tex", "w") as f:
        f.write(full_table_string)
    
    
if __name__ == "__main__":
    fire.Fire(ycbv_pose_tracking_table)
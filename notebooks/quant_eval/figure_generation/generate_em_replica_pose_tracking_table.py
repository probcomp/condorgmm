import pandas as pd
import glob
import condorgmm
import fire

# results_directory = "/home/nishadgothoskar/condorgmm/results/camera_pose_tracking_replica_full"


def generate_em_replica_pose_tracking_table(results_directory=None):
    if results_directory is None:
        results_directory = (
            condorgmm.get_root_path()
            / "assets/condorgmm_bucket/camera_pose_tracking_replica_em_change_constants_2025-03-24-18-44-04"
        )
    results_files = glob.glob(str(results_directory) + "/*.pkl")
    results_dfs = [pd.read_pickle(f) for f in results_files]
    concatenated_df = pd.concat(results_dfs)

    results = {}

    average_ate_on_full_dataset = concatenated_df["value"].mean() * 100.0
    results["Full Dataset"] = average_ate_on_full_dataset

    for scene_name in condorgmm.data.ReplicaVideo.SCENE_NAMES:
        results_df = concatenated_df[concatenated_df["scene"] == scene_name]
        average_ate = results_df["value"].mean() * 100.0
        results[scene_name] = average_ate

    results_string = (
        r"\textbf{condorgmm} & \cellcolor{green}{\textbf{"
        + f"{results['Full Dataset']:.2f}"
        + "}}"
    )

    for scene_name in condorgmm.data.ReplicaVideo.SCENE_NAMES:
        results_string += (
            r" & \cellcolor{green}{\textbf{" + f"{results[scene_name]:.2f}" + "}}"
        )
    results_string += r" & 21.02 \\"

    output_table_string = (
        r"""
\begin{tabular}{lccccccccc|l}
\hline
\addlinespace[4pt]
\textbf{Methods} & \textbf{Avg.} & \texttt{R0} & \texttt{R1} & \texttt{R2} & \texttt{Of0} & \texttt{Of1} & \texttt{Of} & \texttt{Of3} & \texttt{Of4} & FPS\\
\addlinespace[2pt]
\hline
\addlinespace[4pt]
Droid-SLAM & \cellcolor{orange}0.38 &\cellcolor{orange}0.53 &\cellcolor{yellow}0.38 &0.45 & \cellcolor{yellow}0.35 & \cellcolor{yellow}0.24 &\cellcolor{orange}0.36 &\cellcolor{orange}0.33 &\cellcolor{yellow}0.43 & 20.0\\
\addlinespace[.1pt]
\hdashline
\addlinespace[2pt]
Vox-Fusion& 3.09& 1.37& 4.70& 1.47& 8.48& 2.04& 2.58& 1.11& 2.94 & 83.33\\
NICE-SLAM&  1.06& 0.97& 1.31& 1.07& 0.88& 1.00& 1.06& 1.10& 1.13 & 5.64\\
ESLAM& 0.63& 0.71& 0.70& 0.52& 0.57& 0.55& 0.58& 0.72& 0.63 & 5.55\\
Point-SLAM&  0.52& 0.61& 0.41& \cellcolor{orange}0.37& \cellcolor{orange}0.38& 0.48& 0.54& 0.69& 0.72 & 1.18\\
SplaTAM& \cellcolor{yellow}0.36& \cellcolor{yellow}0.31& \cellcolor{orange}0.40& \cellcolor{yellow}0.29& 0.47& \cellcolor{orange}0.27& \cellcolor{yellow}0.29& \cellcolor{yellow}0.32& 0.55 & 1.00\\
"""
        + results_string
        + r""" \bottomrule \end{tabular}"""
    )
    print(output_table_string)

    tex_dir = condorgmm.get_root_path() / "condorgmm_tex"
    with open(tex_dir / "em_replica_pose_tracking_table.tex", "w") as f:
        f.write(output_table_string)


if __name__ == "__main__":
    fire.Fire(generate_em_replica_pose_tracking_table)

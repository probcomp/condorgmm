import json
from collections import defaultdict
from pathlib import Path

import fire
import pandas as pd
from tabulate import tabulate

import condorgmm
from condorgmm.eval.fp_loader import YCBVTrackingResultLoader
from condorgmm.eval.metrics import add_err, adds_err, compute_auc

ALL_METRICS = {
    "ADD": add_err,
    "ADD-S": adds_err,
}


def aggregate_results(metrics_folder, get_by_object, scenes):
    # Initialize metrics dictionary
    all_scores = defaultdict(lambda: defaultdict(list))

    if scenes is not None:
        metrics_files = [
            x
            for scene in scenes
            for x in Path(metrics_folder).glob(
                f"SCENE_{scene}_OBJECT_INDEX_*_METRICS.json"
            )
        ]
    else:
        metrics_files = Path(metrics_folder).glob("SCENE_*_OBJECT_INDEX_*_METRICS.json")

    # Iterate through all metrics files in the folder
    for metrics_file in metrics_files:
        with open(metrics_file, "r") as f:
            single_run_scores = json.load(f)
            # Aggregate the metrics
            for metric_name in single_run_scores:
                for obj_name, values in single_run_scores[metric_name].items():
                    if get_by_object:
                        all_scores[metric_name][obj_name].extend(values)
                    else:
                        all_scores[metric_name][metrics_file.name].extend(values)
                    all_scores[metric_name]["000_ALL_FRAMES"].extend(values)

    # Aggregate results per object and compute AUC
    final_results = {}
    for metric_name in ALL_METRICS:
        final_results[metric_name] = {}
        for obj_name in all_scores[metric_name]:
            final_results[metric_name][obj_name] = float(
                compute_auc(all_scores[metric_name][obj_name])
            )
    return all_scores, final_results


def calculate_metrics(
    metrics_folder,
    baseline=(condorgmm.get_root_path() / "assets/bucket/condorgmm/baseline/metrics"),
    scene=None,
    obj=False,
    output_summary_file=None,
    output_detail_file=None,
):
    if isinstance(scene, tuple):
        scenes = list(range(scene[0], scene[1] + 1))
    elif isinstance(scene, int):
        scenes = [scene]
    elif isinstance(scene, list):
        scenes = scene
    else:
        scenes = None

    print(f"Loading metrics from {metrics_folder} for scenes {scenes} ...")
    metrics_folder = Path(metrics_folder)

    # Aggregate results and compute the final results
    print("Aggregating metrics to compute final results...")
    all_scores, final_results = aggregate_results(metrics_folder, obj, scenes)

    if baseline is not None:
        baseline = str(baseline)
        if not Path(baseline).exists() and baseline.lower().startswith("fp"):
            # parse baseline in the format of "fp-framerate-split". e.g. "fp-50-train_real"
            _, frame_rate, split = baseline.split("-")
            baseline = YCBVTrackingResultLoader(
                frame_rate=int(frame_rate), split=split
            ).metrics_dir
        print(f"Comparing metrics with baseline from {baseline} ...")
        _, baseline_final_results = aggregate_results(baseline, obj, scenes)

        df1 = pd.DataFrame(final_results).sort_index()
        df2 = pd.DataFrame(baseline_final_results).sort_index()

        d = pd.DataFrame()

        d["\u0394ADD"] = df1["ADD"] - df2["ADD"]
        d["\u0394ADD-S"] = df1["ADD-S"] - df2["ADD-S"]
        d["ADD"] = df1["ADD"]
        d["ADD_baseline"] = df2["ADD"]
        d["ADD-S"] = df1["ADD-S"]
        d["ADD-S_baseline"] = df2["ADD-S"]

        def value_to_colored_string(value):
            if value > 0:
                return f"\033[92m+{value:0.4f}\033[0m"
            elif value < 0:
                return f"\033[91m{value:0.4f}\033[0m"
            else:
                return value

        d["\u0394ADD"] = d["\u0394ADD"].map(value_to_colored_string)
        d["\u0394ADD-S"] = d["\u0394ADD-S"].map(value_to_colored_string)
        d = d.map(lambda x: f"{x:0.4f}" if isinstance(x, float) else x)

        print(tabulate(d, headers="keys", tablefmt="psql"))

    else:
        if output_summary_file is None:
            output_summary_file = metrics_folder.parent / "metrics_summary.json"
        if output_detail_file is None:
            output_detail_file = metrics_folder.parent / "metrics_detail.json"

        # Save the final summary results
        with open(output_summary_file, "w") as f:
            json.dump(final_results, f)
        print(f"Final metrics summary saved to {output_summary_file}")

        # Save detailed results
        with open(output_detail_file, "w") as f:
            json.dump(all_scores, f, indent=4)
        print(f"Detailed metrics saved to {output_detail_file}")

        # Print the summary
        print("Final metrics summary:")
        final_results = pd.DataFrame(final_results).sort_index()
        print(tabulate(final_results, headers="keys", tablefmt="psql"))


if __name__ == "__main__":
    fire.Fire(calculate_metrics)

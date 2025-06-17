import condorgmm
import condorgmm.eval.metrics
import condorgmm.data
import datetime as dt
from tqdm import tqdm
from condorgmm import Pose
import fire
import condorgmm.object_tracking.integrated_object_tracking
import condorgmm.object_tracking.em_object_tracking
import condorgmm.object_tracking.low_frame_rate


def run_object_pose_tracking(
    scene=None,
    dataset=None,
    object_index=None,
    experiment_name=None,
    max_T=None,
    mode="integrated",
    debug=False,
    stream_rerun=False,
    downscale=1,
    upscale=1,
    frame_rate=1,
    log_period=1,
):

    if mode == "integrated":
        initialize = condorgmm.object_tracking.integrated_object_tracking.initialize
        update = condorgmm.object_tracking.integrated_object_tracking.update
        rr_log = condorgmm.object_tracking.integrated_object_tracking.rr_log
    elif mode == "em":
        initialize = condorgmm.object_tracking.em_object_tracking.initialize
        update = condorgmm.object_tracking.em_object_tracking.update
        rr_log = condorgmm.object_tracking.em_object_tracking.rr_log
    elif mode == "low_frame_rate":
        initialize = condorgmm.object_tracking.low_frame_rate.initialize
        update = condorgmm.object_tracking.low_frame_rate.update
        rr_log = condorgmm.object_tracking.low_frame_rate.rr_log
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Generate unique run ID using timestamp
    if experiment_name is None:
        experiment_name = ""
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = f"{experiment_name}_{timestamp}"
    print(f"Experiment name: {experiment_name}")

    results_dir = (
        condorgmm.get_root_path()
        / "results"
        / f"object_pose_tracking_{dataset}_{mode}_{experiment_name}"
    )
    output_file = results_dir / "output.txt"

    results_dir.mkdir(parents=True, exist_ok=True)
    # make output file
    with open(output_file, "w") as f:
        f.write(f"Running object pose tracking for {dataset} dataset\n")
        f.write(f"Experiment name: {experiment_name}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Debug: {debug}\n")
        f.write(f"Downscale: {downscale}\n")
        f.write(f"Max T: {max_T}\n")

    if dataset == "ycbineoat":
        DatasetVideoType = condorgmm.data.YCBinEOATVideo
    elif dataset == "ycbv":
        DatasetVideoType = condorgmm.data.YCBVVideo
    elif dataset == "ycbv_test":
        DatasetVideoType = condorgmm.data.YCBTestVideo
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    if scene is None:
        scene_names = DatasetVideoType.SCENE_NAMES
    elif isinstance(scene, str):
        scene_names = [scene]
    elif isinstance(scene, list):
        scene_names = scene
    elif isinstance(scene, int):
        scene_names = [scene]
    else:
        raise ValueError(f"Invalid scene argument: {scene} , type: {type(scene)}")

    print("This will run camera pose tracking for the following scenes:")
    print(scene_names)

    for scene_name in scene_names:
        video = DatasetVideoType(scene_name).downscale(downscale).upscale(upscale)
        max_T_local = min(len(video), max_T) if max_T is not None else len(video)
        timesteps = list(range(0, max_T_local, frame_rate))
        
        print("\tLoading video...")
        
        frames = video.load_frames(timesteps)

        print(f"Running camera pose tracking for YCB-inEOAT dataset scene {scene_name}")
        object_indices = (
            range(len(video[0].object_ids)) if object_index is None else [object_index]
        )
        for current_object_index in object_indices:
            condorgmm.rr_init(f"object pose tracking {scene_name} {current_object_index}")
            print(
                f"Running camera pose tracking for YCB-inEOAT dataset scene {scene_name} and object {current_object_index}"
            )

            print("\tInitializing the object tracker...")

            object_id = video[0].object_ids[current_object_index]
            object_mesh = video.get_object_mesh_from_id(object_id)
            object_name = video.get_object_name_from_id(object_id)

            camera_pose_0, object_pose_0, og_state, debug_data = initialize(
                frame=video[0],
                object_mesh=object_mesh,
                object_idx=current_object_index,
                debug=debug,
            )

            camera_poses_over_time = [camera_pose_0]
            object_poses_over_time = [object_pose_0]

            aggregated_debug_data = []

            state = og_state

            if stream_rerun:
                rr_log(state, frames[timesteps[0]], 0)
            print("\tInitialized.")
            print("\tRunning tracking on all subsequent frames...")

            for timestep in tqdm(timesteps[1:]):
                camera_pose, object_pose, state, debug_data = update(
                    state, video[timestep], timestep=timestep, debug=debug
                )
                camera_poses_over_time.append(camera_pose)
                object_poses_over_time.append(object_pose)
                aggregated_debug_data.append(debug_data)

                if stream_rerun and timestep % log_period == 0:
                    rr_log(state, video[timestep], timestep, do_log_frame=False)



            # Compute metrics
            predicted_poses = [
                camera_poses_over_time[t].inv() @ object_poses_over_time[t]
                for t in range(len(camera_poses_over_time))
            ]
            gt_poses = [
                Pose(video[t].camera_pose).inv()
                @ Pose(video[t].object_poses[current_object_index])
                for t in timesteps
            ]

            vertices = video.get_object_mesh_from_id(
                video[0].object_ids[current_object_index]
            ).vertices

            results_df = condorgmm.eval.metrics.create_empty_results_dataframe()
            condorgmm.eval.metrics.add_object_tracking_metrics_to_results_dataframe(
                results_df,
                scene_name,
                "condorgmm",
                object_name,
                predicted_poses,
                gt_poses,
                vertices,
            )

            # Save metrics to file.
            results_df.to_pickle(
                results_dir
                / f"scene_{scene_name}_object_index_{current_object_index}_object_pose_tracking_results.pkl"
            )

            aggregated_df = condorgmm.eval.metrics.aggregate_dataframe_with_function(
                results_df, condorgmm.eval.metrics.compute_auc
            )
            print_string = f"{aggregated_df}"
            print(print_string)
            with open(output_file, "a") as f:
                f.write(print_string)
                f.write("\n")


if __name__ == "__main__":
    fire.Fire(run_object_pose_tracking)

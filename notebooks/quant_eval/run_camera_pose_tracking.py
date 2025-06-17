import condorgmm
import condorgmm.eval.metrics
import condorgmm.data
from tqdm import tqdm
import fire
import datetime as dt
import importlib
import condorgmm.camera_tracking.integrated_camera_tracking
import condorgmm.camera_tracking.em_camera_tracking
import warp as wp
import rerun as rr


def run_camera_pose_tracking(
    dataset=None,
    scene=None,
    experiment_name=None,
    max_T=None,
    mode="integrated",
    debug=False,
    downscale=4,
    stream_rerun=False,
    save_rerun=False,
    log_period=1,
    condor_log_period=1,
):
    wp.init()

    assert dataset is not None, "Please provide a dataset name (e.g. replica, tum)"

    if mode == "integrated":
        print("Using integrated camera tracking")
        initialize = condorgmm.camera_tracking.integrated_camera_tracking.initialize
        update = condorgmm.camera_tracking.integrated_camera_tracking.update
        rr_log = condorgmm.camera_tracking.integrated_camera_tracking.rr_log
    elif mode == "em":
        print("Using EM camera tracking")
        initialize = condorgmm.camera_tracking.em_camera_tracking.initialize
        update = condorgmm.camera_tracking.em_camera_tracking.update
        rr_log = condorgmm.camera_tracking.em_camera_tracking.rr_log
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
        / f"camera_pose_tracking_{dataset}_{mode}_{experiment_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "replica":
        DatasetVideoType = condorgmm.data.ReplicaVideo
    elif dataset == "tum":
        DatasetVideoType = condorgmm.data.TUMVideo
    elif dataset == "scannet":
        DatasetVideoType = condorgmm.data.ScanNetVideo
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    if scene is None:
        scene_names = DatasetVideoType.SCENE_NAMES
    elif isinstance(scene, str):
        scene_names = [scene]
    elif isinstance(scene, list):
        scene_names = scene
    else:
        raise ValueError(f"Invalid scene argument: {scene}")

    print(
        f"This will run camera pose tracking for the following scenes in the {dataset} dataset:"
    )
    print(scene_names)

    for scene_name in scene_names:
        try:
            video = DatasetVideoType(scene_name).downscale(downscale)
        except Exception as e:
            print(
                f"Failed to load video for {dataset} dataset scene {scene_name}. Got error: {e}"
            )
            continue

        print(f"Running camera pose tracking for {dataset} dataset scene {scene_name}")

        if stream_rerun:
            condorgmm.rr_init(f"camera_tracking-{dataset}-{scene_name}")
        if save_rerun and not stream_rerun:
            rr.init(f"camera_tracking-{dataset}-{scene_name}")
        if save_rerun:
            rr.save(f"{results_dir}/{scene_name}.rrd")

        max_T_local = min(len(video), max_T) if max_T is not None else len(video)
        print("\tLoading video...")
        frames = video.load_frames(range(0, max_T_local, 1))

        aggregated_debug_data = []
        ## Run tracker ##
        print("\tInitializing the camera tracker...")
        camera_pose_0, og_state, debug_data = initialize(
            frames[0], debug=debug, seed=123
        )

        if stream_rerun or save_rerun:
            rr_log(og_state, frames[0], timestep=0)

        state = og_state

        rr_log(state, frames[0], 0)
        print("\tInitialized.")
        print("\tRunning tracking on all subsequent frames...")
        camera_poses_over_time = [camera_pose_0]

        for timestep in tqdm(
            range(1, len(frames)),
        ):
            # Inferring camera pose using gradients
            camera_pose, state, debug_data = update(
                state, frames[timestep], timestep, debug=debug
            )
            camera_poses_over_time.append(camera_pose)
            if (
                stream_rerun
                or save_rerun
                and log_period > 0
            ):
                if timestep % log_period == 0:
                    rr_log(
                        state,
                        frames[timestep],
                        timestep,
                        do_log_frame=True,
                        do_log_condor_state=False,
                    )

            aggregated_debug_data.append(debug_data)
            if not debug_data["gmm_is_valid"]:
                print(f"Invalid GMM at timestep {timestep}")
                break

        # Compute metrics
        importlib.reload(condorgmm.eval.metrics)
        results_df = condorgmm.eval.metrics.create_empty_results_dataframe()

        gt_poses = [condorgmm.Pose(frame.camera_pose) for frame in frames]
        predicted_poses = [
            condorgmm.Pose(camera_poses_over_time[t]) for t in range(len(frames))
        ]

        condorgmm.eval.metrics.add_camera_tracking_metrics_to_results_dataframe(
            results_df,
            scene_name,
            "condorgmm",
            predicted_poses,
            gt_poses,
        )

        # Save metrics to file.
        results_df.to_pickle(
            results_dir / f"{scene_name}_camera_pose_tracking_results.pkl"
        )
        print(
            condorgmm.eval.metrics.aggregate_dataframe_with_function(
                results_df, lambda x: x.mean()
            )
        )


if __name__ == "__main__":
    fire.Fire(run_camera_pose_tracking)

import pytest

from condorgmm.eval.fp_loader import (
    FP_RESULTS_ROOT_DIR,
    YCBVTrackingResultLoader,
)

if not FP_RESULTS_ROOT_DIR.exists():
    pytest.skip(
        "No foundation pose tracking results found. Please run "
        "`python scripts/bucket_utils/pull.py ` to fetch the precomputed results.",
        allow_module_level=True,
    )


def test_loading_precomputed_ycbv_results_training_real():
    fp_ycbv_result = YCBVTrackingResultLoader(frame_rate=50, split="train_real")

    precomputed_poses = fp_ycbv_result.get_poses(test_scene_id=3, object_id=1)
    assert precomputed_poses.shape == (46, 4, 4)

    test_scenes = fp_ycbv_result.get_scene_ids()
    assert len(test_scenes) == 80

    obj_ids_scene_2 = fp_ycbv_result.get_object_ids(test_scene_id=2)
    assert len(obj_ids_scene_2) == 6

    precomputed_metrics = fp_ycbv_result.get_metrics(test_scene_id=3, object_id=1)
    assert "ADD" in precomputed_metrics
    assert "ADD-S" in precomputed_metrics
    assert "009_gelatin_box" in precomputed_metrics["ADD"]
    # there are 46 frames in total, but the first frame is not scored
    assert len(precomputed_metrics["ADD"]["009_gelatin_box"]) == 46 - 1

    # results on original frame rates
    fp_ycbv_result = YCBVTrackingResultLoader(frame_rate=1, split="train_real")

    precomputed_poses = fp_ycbv_result.get_poses(test_scene_id=3, object_id=1)
    assert precomputed_poses.shape == (2298, 4, 4)

    precomputed_metrics = fp_ycbv_result.get_metrics(test_scene_id=3, object_id=1)
    # the first frame is not scored
    assert len(precomputed_metrics["ADD-S"]["009_gelatin_box"]) == 2298 - 1


def test_loading_precomputed_ycbv_results_test():
    fp_ycbv_result = YCBVTrackingResultLoader(frame_rate=50, split="test")

    precomputed_poses = fp_ycbv_result.get_poses(test_scene_id=48, object_id=1)
    assert precomputed_poses.shape == (45, 4, 4)

    test_scenes = fp_ycbv_result.get_scene_ids()
    assert len(test_scenes) == 12

    obj_ids_scene_48 = fp_ycbv_result.get_object_ids(test_scene_id=48)
    assert len(obj_ids_scene_48) == 5

    precomputed_metrics = fp_ycbv_result.get_metrics(test_scene_id=48, object_id=1)
    assert "ADD" in precomputed_metrics
    assert "ADD-S" in precomputed_metrics
    assert "007_tuna_fish_can" in precomputed_metrics["ADD"]
    # there are 45 frames in total, but the first frame is not scored
    assert len(precomputed_metrics["ADD"]["007_tuna_fish_can"]) == 45 - 1

    # results on original frame rates
    fp_ycbv_result = YCBVTrackingResultLoader(frame_rate=1, split="test")

    precomputed_poses = fp_ycbv_result.get_poses(test_scene_id=48, object_id=1)
    assert precomputed_poses.shape == (2243, 4, 4)

    precomputed_metrics = fp_ycbv_result.get_metrics(test_scene_id=48, object_id=1)
    # first frame is not scored
    assert len(precomputed_metrics["ADD"]["007_tuna_fish_can"]) == 2243 - 1


def test_invalid_configs():
    with pytest.raises(ValueError):
        # we do not have precomputed results for this config
        YCBVTrackingResultLoader(frame_rate=3, split="train")

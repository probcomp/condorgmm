import os
import pytest
import condorgmm
import condorgmm.data
import numpy as np


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_ycb_dataloader():
    dataset = condorgmm.data.YCBVVideo(1)
    img0 = dataset[0]
    imglast = dataset[len(dataset) - 1]
    assert np.allclose(img0.intrinsics, imglast.intrinsics)

    downscaled = img0.downscale(2)
    assert np.all(downscaled.rgb == dataset.downscale(2)[0].rgb)
    assert img0.height == 2 * downscaled.height
    assert img0.width == 2 * downscaled.width

    cropped = img0.crop(10, 20, 10, 20)
    assert np.all(cropped.rgb == dataset.crop(10, 20, 10, 20)[0].rgb)
    assert cropped.height == 10
    assert cropped.width == 10


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_r3d_dataloader():
    r3d_path = condorgmm.get_root_path() / "assets/bucket/input_data/wireless_charger.r3d"
    dataset = condorgmm.data.R3DVideo(r3d_path)
    img0 = dataset[0]
    imglast = dataset[len(dataset) - 1]
    assert np.allclose(img0.intrinsics, imglast.intrinsics)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_tum_dataloader():
    scene = "freiburg2_xyz"
    dataset = condorgmm.data.TUMVideo(scene)
    img0 = dataset[0]
    imglast = dataset[len(dataset) - 1]
    assert np.allclose(img0.intrinsics, imglast.intrinsics)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_replica_dataloader():
    scene = "office1"
    dataset = condorgmm.data.ReplicaVideo(scene)
    img0 = dataset[0]
    imglast = dataset[len(dataset) - 1]
    assert np.allclose(img0.intrinsics, imglast.intrinsics)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_ycbineoat_dataloader():
    dataset = condorgmm.data.YCBinEOATVideo("bleach0")
    img0 = dataset[0]
    imglast = dataset[len(dataset) - 1]
    assert np.allclose(img0.intrinsics, imglast.intrinsics)
    dataset.get_object_mesh_from_id(dataset[0].object_ids[0])


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_scannet_dataloader():
    dataset = condorgmm.data.ScanNetVideo(0)
    img0 = dataset[0]
    imglast = dataset[len(dataset) - 1]
    assert np.allclose(img0.intrinsics, imglast.intrinsics)

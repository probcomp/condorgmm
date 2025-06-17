import jax.numpy as jnp
import condorgmm
import condorgmm.data as data
import condorgmm.condor.interface.camera_tracking as ct
from condorgmm.condor.types import FloatFromDiscreteSet


def test_camera_tracking():
    video = data.YCBVVideo.training_scene(2).downscale(4)

    frame = video[0]
    n_gaussians = 100
    camera_pose_world_frame = condorgmm.Pose(video[0].camera_pose)

    gmm, ccts, meta = ct.initialize(  # type: ignore
        frame,
        camera_pose_world_frame,
        ct.fast_config.replace(n_gaussians=n_gaussians),
        log=True,
    )

    assert ccts.state.gaussian_has_assoc_mask.sum() > n_gaussians * 0.8

    # Test that the range of global parameters is not too small, relative to
    # the parameter range that inference wants to use.  Specifically, test
    # that inference never drove any global paramters to the very top or bottom
    # of the range.
    visited_global_param_values = (
        meta.visited_states.states.matter.background_initialization_params  # type: ignore
    )
    for param_name, value in visited_global_param_values.__dict__.items():
        if isinstance(value, FloatFromDiscreteSet):
            dom_size = len(value.domain)
            assert jnp.all(
                value.idx != 0
            ), f"Parameter {param_name} hit the bottom of its range during inference."
            assert jnp.all(
                value.idx != dom_size - 1
            ), f"Parameter {param_name} hit the top of its range during inference."

    gmm1, ccts1 = ct.update(  # type: ignore
        video[1],
        condorgmm.Pose(video[1].camera_pose),
        ccts,
        ct.fast_config.replace(n_gaussians=n_gaussians),
    )

    st = ccts1.state
    persisting_gaussians_with_assoc = (
        st.gaussians[st.gaussian_has_assoc_mask].origin > -1
    ).sum()
    assert persisting_gaussians_with_assoc > 60

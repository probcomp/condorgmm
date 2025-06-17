import jax
from jax.random import key as prngkey
import jax.numpy as jnp
from condorgmm.condor.pose import Pose
import condorgmm.condor.model.model as model
from condorgmm.condor.config import DEFAULT_HYPERPARAMS
from condorgmm.condor.types import (
    CondorGMMState,
    MyPytree,
    SingleKnownObjectSceneState,
    BackgroundGaussianEvolutionParams,
    FloatArray,
)
import condorgmm
from condorgmm.condor.rerun import log_state

init_model_simulate = jax.jit(model.init_model.simulate)
step_jit = jax.jit(model.step_model.simulate)


def test_frame0_simulate():
    hypers = DEFAULT_HYPERPARAMS.replace(
        n_gaussians=20,
        datapoint_mask=jnp.ones(100, dtype=bool),
        running_simulate=jnp.array(True, dtype=bool),
    )
    tr = init_model_simulate(prngkey(0), (hypers,))
    assert tr is not None
    assert isinstance(tr.get_retval(), CondorGMMState)

    ## Run this to visualize the generated state:
    # import condorgmm
    # from condorgmm.condor.rerun import log_state
    # condorgmm.rr_init("condor2_sim_00")
    # log_state(tr.get_retval(), hypers)


def test_simulate_with_object():
    # Generate an object model
    hypers = DEFAULT_HYPERPARAMS.replace(
        n_gaussians=10,
        datapoint_mask=jnp.ones(1, dtype=bool),
        running_simulate=jnp.array(True, dtype=bool),
    )
    tr = model.init_model.simulate(prngkey(1), (hypers,))
    gaussians = tr.get_retval().gaussians

    # Generate a condorgmmScene with this as the object model
    scene = SingleKnownObjectSceneState(
        transform_World_Camera=Pose.identity(),
        transform_World_Object=Pose(
            jnp.array([500, 500, 500.0]), jnp.array([0.0, 0.0, 0.0, 1.0])
        ),
        object_model=gaussians,
    )
    hypers = DEFAULT_HYPERPARAMS.replace(
        n_gaussians=100,
        initial_scene=scene,
        datapoint_mask=jnp.ones(1000, dtype=bool),
        running_simulate=jnp.array(True, dtype=bool),
        default_background_evolution_params=BackgroundGaussianEvolutionParams[
            FloatArray
        ](
            prob_gaussian_is_new=jnp.array(0.01, dtype=jnp.float32),
            xyz_cov_pcnt=jnp.array(5.0, dtype=jnp.float32),
            rgb_var_pcnt=jnp.array(40.0, dtype=jnp.float32),
            target_xyz_mean_std=jnp.array(0.003, dtype=jnp.float32),
        ),
    )
    tr = model.init_model.simulate(prngkey(0), (hypers,))
    assert tr is not None
    assert isinstance(tr.get_retval(), CondorGMMState)

    import rerun as rr

    condorgmm.rr_init("objmodel_01")
    rr.set_time_sequence("frame", 0)
    log_state(tr.get_retval(), hypers)

    hyp2 = hypers.replace(object_pose_drift_std=jnp.array(100, dtype=float))

    st = tr.get_retval()
    for i in range(1, 2):
        st = step_jit(prngkey(i), (st, hyp2)).get_retval()
        assert isinstance(st, CondorGMMState)
        rr.set_time_sequence("frame", i)
        log_state(st, hyp2)


def test_step_simulate():
    hypers = DEFAULT_HYPERPARAMS.replace(
        n_gaussians=20,
        datapoint_mask=jnp.ones(100, dtype=bool),
        running_simulate=jnp.array(True, dtype=bool),
    )
    tr0 = init_model_simulate(prngkey(0), (hypers,))
    tr1 = step_jit(prngkey(0), (tr0.get_retval(), hypers))
    assert tr1 is not None
    assert isinstance(tr1.get_retval(), CondorGMMState)


## These next tests are not integration tests, but I put them here
# because some of them simulate from the whole model as a starting point,
# meaning that these tests will be as slow as an integration test of
# the model.


def test_origin_to_choicemap():
    hypers = DEFAULT_HYPERPARAMS.replace(
        n_gaussians=20,
        datapoint_mask=jnp.ones(100, dtype=bool),
        running_simulate=jnp.array(True, dtype=bool),
    )

    for o in [-1, 10]:
        assert (
            o
            == model.generate_origin.importance(
                prngkey(0),
                model.origin_to_choicemap(jnp.array(o)),
                (o, hypers.default_background_evolution_params, hypers),
            )[0].get_retval()
        )


def test_gaussian_to_choicemap():
    hypers = DEFAULT_HYPERPARAMS.replace(
        n_gaussians=20,
        datapoint_mask=jnp.ones(100, dtype=bool),
        running_simulate=jnp.array(True, dtype=bool),
    )
    st0: model.CondorGMMState = model.init_model.simulate(
        prngkey(0), (hypers,)
    ).get_retval()
    st1 = model.step_model.simulate(prngkey(0), (st0, hypers)).get_retval()
    for origin in [-1, 0]:
        assert MyPytree.eq(
            st1.gaussians[0].replace(origin=origin),
            model.generate_gaussian_i_at_noninitial_timestep.importance(
                prngkey(0),
                model.gaussian_to_choicemap_for_ggiant(
                    st1.gaussians[0].replace(origin=origin)
                ),
                (
                    0,
                    st1.scene,
                    st0.scene,
                    st0.matter,
                    st0.gaussian_has_assoc_mask,
                    hypers,
                ),
            )[0].get_retval(),
        )

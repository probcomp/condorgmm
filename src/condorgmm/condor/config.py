import jax.numpy as jnp
from .types import (
    Pose,
    Hyperparams,
    BackgroundOnlySceneState,
    NewGaussianPriorParams,
    NewGaussianPriorParamsDomains,
    Intrinsics,
    Domain,
    FloatFromDiscreteSet,
    BackgroundGaussianEvolutionParams,
    FloatArray,
    EvolvedGaussianPriorParamsDomains,
)

STD_PSEUDOCOUNT_DOMAIN_1D = Domain(jnp.logspace(base=10, start=-4, stop=3, num=64))
COV_PSEUDOCOUNT_DOMAIN_3D = Domain(2 + jnp.logspace(base=10, start=-4, stop=3, num=64))
MEAN_PSEUDOCOUNT_DOMAIN = Domain(jnp.logspace(base=10, start=-8, stop=2, num=64))
RGB_PSEUDO_STD_DOMAIN = Domain(jnp.logspace(base=10, start=-3, stop=3.5, num=64))
XYZ_PSEUDO_STD_DOMAIN = Domain(jnp.logspace(base=10, start=-5, stop=1, num=64))

# Domain for probability of new Gaussian, logarithmically spaced between 1e-8 and 0.9
PROB_NEW_GAUSSIAN_DOMAIN = Domain(
    jnp.logspace(base=10, start=-8, stop=jnp.log10(0.9), num=64)
)


def _i(x):
    return jnp.array(x, dtype=jnp.int32)


def _f(x):
    return jnp.array(x, dtype=jnp.float32)


def _b(x):
    return jnp.array(x, dtype=jnp.bool_)


prior_param_domains = NewGaussianPriorParamsDomains(
    std_pseudocount_domain_1d=STD_PSEUDOCOUNT_DOMAIN_1D,
    cov_pseudocount_domain_3d=COV_PSEUDOCOUNT_DOMAIN_3D,
    mean_pseudocount_domain=MEAN_PSEUDOCOUNT_DOMAIN,
    rgb_pseudo_std_domain=RGB_PSEUDO_STD_DOMAIN,
    xyz_pseudo_std_domain=XYZ_PSEUDO_STD_DOMAIN,
)

evolved_gaussian_prior_param_domains = EvolvedGaussianPriorParamsDomains(
    prob_gaussian_is_new_domain=PROB_NEW_GAUSSIAN_DOMAIN,
    xyz_cov_pcnt_domain=COV_PSEUDOCOUNT_DOMAIN_3D,
    rgb_var_pcnt_domain=STD_PSEUDOCOUNT_DOMAIN_1D,
    target_xyz_mean_std_domain=XYZ_PSEUDO_STD_DOMAIN,
)


def initial_new_gaussian_prior_params() -> NewGaussianPriorParams[FloatFromDiscreteSet]:
    ## RGB
    # Here's an alpha/beta and 1/beta I found visually that
    # leads to a broad distribution with something like
    # 90% mass on std in [1, 100].
    rgb_var_pseudo_sample_std = jnp.array(2.51, dtype=jnp.float32)
    rgb_var_pseudo_sample_var = rgb_var_pseudo_sample_std**2
    n_pseudo_observations = jnp.array(0.8, dtype=jnp.float32)
    # Set this so that the std of the color means should be around 255/2.
    rgb_mean_n_pseudo_obs = rgb_var_pseudo_sample_var / (128**2)

    ## XYZ
    xyz_cov_n_pseudo_obs = jnp.array(2.1, dtype=jnp.float32)
    xyz_cov_isotropic_pseudo_sample_stds = (
        jnp.array([5, 5, 5], dtype=jnp.float32) * 1e-3
    )
    xyz_mean_n_pseudo_obs = jnp.array(0.00001, dtype=jnp.float32)

    return NewGaussianPriorParams[jnp.ndarray](
        xyz_cov_pcnt=xyz_cov_n_pseudo_obs,
        xyz_cov_isotropic_prior_stds=xyz_cov_isotropic_pseudo_sample_stds,
        xyz_mean_pcnt=xyz_mean_n_pseudo_obs,
        rgb_var_n_pseudo_obs=n_pseudo_observations * jnp.ones(3, dtype=jnp.float32),
        rgb_var_pseudo_sample_stds=rgb_var_pseudo_sample_std
        * jnp.ones(3, dtype=jnp.float32),
        rgb_mean_n_pseudo_obs=rgb_mean_n_pseudo_obs * jnp.ones(3, dtype=jnp.float32),
    ).discretize(prior_param_domains)


DEFAULT_HYPERPARAMS = Hyperparams(
    # This group of params must be overridden by user
    n_gaussians=0,
    datapoint_mask=jnp.array([]),
    intrinsics=Intrinsics(_f(0), _f(0), _f(0), _f(0), _f(0), _f(0), 0, 0),
    # Tiling hypers
    use_monolithic_tiling=False,
    tile_size_x=16,
    tile_size_y=16,
    max_n_gaussians_per_tile=16,
    # Hypers for generate_initial_scene
    initial_scene=BackgroundOnlySceneState(
        transform_World_Camera=Pose.identity(),
    ),
    # Hypers for generate_new_gaussian
    prior_param_domains=prior_param_domains,
    evolved_gaussian_prior_param_domains=evolved_gaussian_prior_param_domains,
    initial_crp_alpha_background=_f(3.0),
    initial_crp_alpha_object=_f(
        40.0
    ),  # prefer object gaussians, and prefer spreading out weight on many of them
    # Hypers for generate_datapoints
    p_depth_nonreturn=_f(0.01),
    # Hypers for step model
    default_background_evolution_params=BackgroundGaussianEvolutionParams[FloatArray](
        prob_gaussian_is_new=_f(0.01),
        xyz_cov_pcnt=_f(50.0),
        rgb_var_pcnt=_f(40.0),
        target_xyz_mean_std=_f(0.0005),  # in meters, higher = more motion allowed
    ).discretize(evolved_gaussian_prior_param_domains),
    camera_pose_drift_std=_f(0.01),  # will change depending on camera speed
    camera_pose_drift_concentration=_f(4000.0),  # will change depending on camera speed
    object_pose_drift_std=_f(0.01),
    object_pose_drift_concentration=_f(4000.0),
    xyz_cov_evolution_pcnt_object=_f(100.0),
    xyz_mean_evolution_pcnt_object=_f(100.0),
    rgb_var_evolution_pcnt_object=_f(10.0),
    target_rgb_mean_variance_for_object_evolution=_f(20.0) ** 2,
    alpha_multiplier_for_evolved_gaussian=_f(5.0),
    crp_alpha_for_new_background_gaussian_in_step_model=_f(5.0),
    xyz_cov_pcnt_object_initialization=_f(100.0),
    xyz_mean_pcnt_object_initialization=_f(100.0),
    rgb_var_pcnt_object_initialization=_f(6.0),
    target_rgb_mean_variance_object_initialization=_f(32.0) ** 2,
    n_unobserved_frames_to_object_gaussian_reset=_i(4),
    ## Inference hyperparams ##
    always_accept_assoc_depth_move=True,
    initial_new_gaussian_prior_params=initial_new_gaussian_prior_params(),
    repopulate_depth_nonreturns=True,
    infer_background_evolution_params=True,
    do_pose_update=False,
    ## Misc. ##
    running_simulate=False,
    rgb_noisefloor_std=_f(0.5),
)

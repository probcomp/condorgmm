import jax
import jax.numpy as jnp
from typing import TypeVar, Self
from genjax import Pytree, Mask
from abc import abstractmethod
from ..pose import Pose
from ..utils import MyPytree
from ..geometry import (
    xyz_from_cameraxyd,
    xyz_to_cameraxyd,
    isovars_and_quaternion_to_cov,
    cov_to_isovars_and_quaternion,
)
from .camera import Intrinsics
from .discrete_floats import Domain, FloatFromDiscreteSet
from genjax.typing import BoolArray, FloatArray, IntArray

# Global constant needed in the condorgmm model and inference.
# Any positive value is fine.
GAMMA_RATE_PARAMETER = 1.0


@Pytree.dataclass
class Gaussian(MyPytree):
    idx: IntArray
    xyz: FloatArray  # (3,) xyz mean in camera frame
    xyz_cov: FloatArray  # (3, 3)
    rgb: FloatArray  # (3,); floats in [0, 255]
    rgb_vars: FloatArray  # (3,)
    mixture_weight: FloatArray
    origin: IntArray
    object_idx: IntArray
    n_frames_since_last_had_assoc: IntArray
    # = -1 if it never had an assoc before
    # = 1 if it had an assoc at the last frame
    # = 2 if it had an assoc 2 frames ago
    # = k if it had an assoc k frames ago.
    # Whether this currently has an assoc does not matter for this field.
    # Identity is carried through the `origin` field.

    def __post_init__(self):
        assert self.rgb.dtype == jnp.float32

    def has_nan(self):
        return jnp.any(
            jnp.array(
                [
                    jnp.isnan(self.xyz).any(),
                    jnp.isnan(self.xyz_cov).any(),
                    jnp.isnan(self.rgb).any(),
                    jnp.isnan(self.rgb_vars).any(),
                    jnp.isnan(self.mixture_weight),
                ]
            )
        )

    def has_extreme_value(self):
        extreme_rgb = jnp.any(jnp.logical_or(self.rgb < -1e4, self.rgb > 255 + 1e4))
        extreme_rgb_vars = jnp.any(self.rgb_vars > 1e4**2)
        extreme_xyz = jnp.any(jnp.logical_or(self.xyz < -1e4, self.xyz > 1e4))
        extreme_xyz_cov = jnp.any(self.xyz_cov > 1e4**2)
        return jnp.any(
            jnp.array(
                [
                    extreme_rgb,
                    extreme_rgb_vars,
                    extreme_xyz,
                    extreme_xyz_cov,
                ]
            )
        )

    def is_valid(self):
        return jnp.logical_and(
            jnp.logical_not(self.has_nan()),
            jnp.logical_not(self.has_extreme_value()),
        )

    def transform_by(self, transform: Pose) -> Self:
        if self.xyz.ndim == 1:
            return _transform_by_single(self, transform)  # type: ignore
        else:
            return jax.vmap(_transform_by_single, in_axes=(0, None))(self, transform)  # type: ignore


def _transform_by_single(gaussian: Gaussian, transform: Pose) -> Gaussian:
    isovars, quat = cov_to_isovars_and_quaternion(gaussian.xyz_cov)
    current_pose = Pose(gaussian.xyz, quat)
    new_pose = transform @ current_pose
    new_cov = isovars_and_quaternion_to_cov(isovars, new_pose.quaternion)
    return gaussian.replace(
        xyz=new_pose.pos,
        xyz_cov=new_cov,
    )


@Pytree.dataclass
class NIWParams(MyPytree):
    cov_pcnt: FloatArray  # pseudo-count for covariance
    prior_cov: FloatArray  # prior covariance
    mean_pcnt: FloatArray  # pseudo-count for mean
    prior_mean: FloatArray  # prior mean


@Pytree.dataclass
class NIGParams(MyPytree):
    var_pcnt: FloatArray  # ()
    prior_var: FloatArray  # ()
    mean_pcnt: FloatArray  # ()
    prior_mean: FloatArray  # ()


@Pytree.dataclass
class NewGaussianPriorParamsDomains(MyPytree):
    std_pseudocount_domain_1d: Domain
    cov_pseudocount_domain_3d: Domain
    mean_pseudocount_domain: Domain
    rgb_pseudo_std_domain: Domain
    xyz_pseudo_std_domain: Domain


@Pytree.dataclass
class EvolvedGaussianPriorParamsDomains(MyPytree):
    prob_gaussian_is_new_domain: Domain  # Domain for probability of a new Gaussian
    xyz_cov_pcnt_domain: Domain  # Domain for xyz covariance evolution pseudo-count
    rgb_var_pcnt_domain: Domain  # Domain for RGB variance evolution pseudo-count
    target_xyz_mean_std_domain: (
        Domain  # Domain for expected translational motion std (in meters)
    )


T = TypeVar("T", FloatArray, FloatFromDiscreteSet)


@Pytree.dataclass
class NewGaussianPriorParams[T](MyPytree):
    xyz_cov_pcnt: T  # (1,)
    xyz_cov_isotropic_prior_stds: T  # (3,)
    xyz_mean_pcnt: T  # (1,)

    rgb_var_n_pseudo_obs: T  # (3,)
    rgb_var_pseudo_sample_stds: T  # (3,)
    rgb_mean_n_pseudo_obs: T  # (3,)

    @property
    def values(self) -> "NewGaussianPriorParams[jnp.ndarray]":
        def to_value(x):
            if isinstance(x, FloatFromDiscreteSet):
                return x.value
            return x

        return NewGaussianPriorParams(
            **{key: to_value(x) for key, x in self.__dict__.items()}
        )

    def discretize(
        self: "NewGaussianPriorParams[jnp.ndarray]", doms: NewGaussianPriorParamsDomains
    ) -> "NewGaussianPriorParams[FloatFromDiscreteSet]":
        def disc(x, dom):
            assert isinstance(x, jnp.ndarray)
            if x.shape == ():
                return dom.first_value_above(x)
            return jax.vmap(lambda x: dom.first_value_above(x))(x)

        return NewGaussianPriorParams(
            disc(self.xyz_cov_pcnt, doms.cov_pseudocount_domain_3d),
            disc(self.xyz_cov_isotropic_prior_stds, doms.xyz_pseudo_std_domain),
            disc(self.xyz_mean_pcnt, doms.mean_pseudocount_domain),
            disc(self.rgb_var_n_pseudo_obs, doms.std_pseudocount_domain_1d),
            disc(self.rgb_var_pseudo_sample_stds, doms.rgb_pseudo_std_domain),
            disc(self.rgb_mean_n_pseudo_obs, doms.mean_pseudocount_domain),
        )

    @property
    def xyz_cov_isotropic_prior_vars(self):
        return self.values.xyz_cov_isotropic_prior_stds**2

    @property
    def rgb_var_pseudo_sample_vars(self):
        return self.values.rgb_var_pseudo_sample_stds**2

    @property
    def xyz_prior_cov(self):
        return jnp.eye(3) * self.values.xyz_cov_isotropic_prior_vars

    def xyz_params(self, xyz_prior_mean: jnp.ndarray) -> NIWParams:
        return NIWParams(
            cov_pcnt=self.values.xyz_cov_pcnt,
            prior_cov=self.values.xyz_prior_cov,
            mean_pcnt=self.values.xyz_mean_pcnt,
            prior_mean=xyz_prior_mean,
        )

    @property
    def rgb_params(self) -> NIGParams:
        return NIGParams(
            var_pcnt=self.values.rgb_var_n_pseudo_obs,
            prior_var=self.values.rgb_var_pseudo_sample_vars,
            mean_pcnt=self.values.rgb_mean_n_pseudo_obs,
            prior_mean=self.values.rgb_mean_center,
        )

    @property
    def rgb_mean_center(self):
        return jnp.array([255 / 2, 255 / 2, 255 / 2], dtype=jnp.float32)


@Pytree.dataclass
class BackgroundGaussianEvolutionParams[T](MyPytree):
    prob_gaussian_is_new: T  # prob_background_gaussian_is_new
    xyz_cov_pcnt: T  # xyz_cov_evolution_pcnt_background
    rgb_var_pcnt: T  # rgb_var_evolution_pcnt_background
    target_xyz_mean_std: T  # standard deviation (in meters) of expected translational motion of each Gaussian relative to the background, from frame to frame

    @property
    def values(self) -> "BackgroundGaussianEvolutionParams[jnp.ndarray]":
        def to_value(x):
            if isinstance(x, FloatFromDiscreteSet):
                return x.value
            return x

        return BackgroundGaussianEvolutionParams(
            **{key: to_value(x) for key, x in self.__dict__.items()}
        )

    def discretize(
        self: "BackgroundGaussianEvolutionParams[jnp.ndarray]",
        doms: EvolvedGaussianPriorParamsDomains,
    ) -> "BackgroundGaussianEvolutionParams[FloatFromDiscreteSet]":
        def disc(x, dom):
            assert isinstance(x, jnp.ndarray)
            return dom.first_value_above(x)

        return BackgroundGaussianEvolutionParams(
            prob_gaussian_is_new=disc(
                self.prob_gaussian_is_new, doms.prob_gaussian_is_new_domain
            ),
            xyz_cov_pcnt=disc(self.xyz_cov_pcnt, doms.xyz_cov_pcnt_domain),
            rgb_var_pcnt=disc(self.rgb_var_pcnt, doms.rgb_var_pcnt_domain),
            target_xyz_mean_std=disc(
                self.target_xyz_mean_std, doms.target_xyz_mean_std_domain
            ),
        )


@Pytree.dataclass
class Tiling(MyPytree):
    @abstractmethod
    def relevant_datapoints_for_gaussian(self, gaussian_idx: int) -> Mask[int]:
        pass

    @abstractmethod
    def relevant_gaussians_for_datapoint(self, datapoint_idx: int) -> Mask[int]:
        pass

    @abstractmethod
    def gaussian_is_relevant_for_datapoint(
        self, gaussian_idx: int, datapoint_idx: int
    ) -> bool:
        pass

    @abstractmethod
    def update_tiling(self, gaussians: Gaussian, key) -> Self:
        pass


@Pytree.dataclass
class VisualMatter(MyPytree):
    background_initialization_params: NewGaussianPriorParams[FloatFromDiscreteSet]
    background_evolution_params: BackgroundGaussianEvolutionParams[FloatFromDiscreteSet]
    gaussians: Gaussian  # batched
    tiling: Tiling

    @property
    def probs(self):
        weights = self.gaussians.mixture_weight
        return weights / jnp.sum(weights)


@Pytree.dataclass
class SceneState(MyPytree):
    # All scene states are expected to have this property.
    transform_World_Camera: Pose


@Pytree.dataclass
class BackgroundOnlySceneState(SceneState):
    transform_World_Camera: Pose


@Pytree.dataclass
class SingleKnownObjectSceneState(SceneState):
    transform_World_Camera: Pose
    transform_World_Object: Pose
    object_model: Gaussian


@Pytree.dataclass
class Observation(MyPytree):
    rgb: jnp.ndarray
    camera_xy: jnp.ndarray
    depth: jnp.ndarray  # 0 = depth non-return

    def __post_init__(self):
        assert self.rgb.dtype == jnp.float32


@Pytree.dataclass
class Datapoint(MyPytree):
    obs: Observation

    # Latents:
    xyz: FloatArray  # in camera frame.  xy here does not equal camera_xy.
    gaussian_idx: IntArray

    @classmethod
    def from_obs_det(cls, obs: Observation, gaussian_idx: IntArray, hypers) -> Self:
        # Fills in xyz even if depth == 0 (ie. even if this is a depth nonreturn).
        xyd = jnp.concatenate([obs.camera_xy, jnp.array([obs.depth])])
        xyz = xyz_from_cameraxyd(xyd, hypers.intrinsics)
        return cls(obs, xyz, gaussian_idx)

    @classmethod
    def from_xyz_rgb(
        cls, xyz: jnp.ndarray, rgb: jnp.ndarray, gaussian_idx: IntArray, hypers
    ) -> Self:
        camera_xyd = xyz_to_cameraxyd(xyz, hypers.intrinsics)
        obs = Observation(rgb, camera_xyd[:2], camera_xyd[2])
        return cls(obs, xyz, gaussian_idx)

    @property
    def rgb(self):
        return self.obs.rgb

    @property
    def camera_xy(self):
        return self.obs.camera_xy


@Pytree.dataclass
class CondorGMMState(MyPytree):
    # The model currently supports 2 types of `SceneState`, and explicitly
    # branches on the type:
    scene: BackgroundOnlySceneState | SingleKnownObjectSceneState
    matter: VisualMatter
    datapoints: Mask[Datapoint]  # batched

    @property
    def gaussians(self):
        return self.matter.gaussians

    @property
    def gaussian_has_assoc_mask(self):
        def has_assoc(gidx):
            masked_dp_idxs = self.matter.tiling.relevant_datapoints_for_gaussian(gidx)
            unmasked_datapoints = self.datapoints.value[masked_dp_idxs.value]
            overall_mask = jnp.logical_and(
                self.datapoints.flag[masked_dp_idxs.value],  # type: ignore
                masked_dp_idxs.flag,  # type: ignore
            )
            return jnp.any(
                jnp.logical_and(unmasked_datapoints.gaussian_idx == gidx, overall_mask)
            )

        return jax.vmap(has_assoc)(jnp.arange(len(self.gaussians)))

    @property
    def gaussians_with_assoc(self) -> Mask[Gaussian]:
        return Mask(
            self.matter.gaussians,
            self.gaussian_has_assoc_mask,
        )

    @property
    def n_assocs_per_gaussian(self):
        def n_assocs(gidx):
            masked_dp_idxs = self.matter.tiling.relevant_datapoints_for_gaussian(gidx)
            unmasked_datapoints = self.datapoints.value[masked_dp_idxs.value]
            overall_mask = jnp.logical_and(
                self.datapoints.flag[masked_dp_idxs.value],  # type: ignore
                masked_dp_idxs.flag,  # type: ignore
            )
            return jnp.sum(
                jnp.logical_and(unmasked_datapoints.gaussian_idx == gidx, overall_mask)
            )

        return jax.vmap(n_assocs)(jnp.arange(len(self.gaussians)))


@Pytree.dataclass
class Hyperparams(MyPytree):
    # This group of params must be overridden by user
    n_gaussians: int = Pytree.static()
    datapoint_mask: BoolArray = (
        Pytree.field()
    )  # Controls how many datapoints to generate.
    intrinsics: Intrinsics = Pytree.field()

    # Tiling hypers
    use_monolithic_tiling: bool = Pytree.static()
    tile_size_x: int = Pytree.static()
    tile_size_y: int = Pytree.static()
    max_n_gaussians_per_tile: int = Pytree.static()

    # Hypers for generate_initial_scene
    initial_scene: BackgroundOnlySceneState | SingleKnownObjectSceneState = (
        Pytree.field()
    )

    # Hypers for generate_new_gaussian
    prior_param_domains: NewGaussianPriorParamsDomains = Pytree.field()
    evolved_gaussian_prior_param_domains: EvolvedGaussianPriorParamsDomains = (
        Pytree.field()
    )
    initial_crp_alpha_background: FloatArray = Pytree.field()
    initial_crp_alpha_object: FloatArray = Pytree.field()

    # Hypers for generate_datapoints
    p_depth_nonreturn: FloatArray = Pytree.field()

    # Hypers for step model
    default_background_evolution_params: BackgroundGaussianEvolutionParams[
        FloatFromDiscreteSet
    ] = Pytree.field()
    camera_pose_drift_std: FloatArray = Pytree.field()
    camera_pose_drift_concentration: FloatArray = Pytree.field()
    object_pose_drift_std: FloatArray = Pytree.field()
    object_pose_drift_concentration: FloatArray = Pytree.field()
    xyz_cov_evolution_pcnt_object: FloatArray = Pytree.field()
    xyz_mean_evolution_pcnt_object: FloatArray = Pytree.field()
    rgb_var_evolution_pcnt_object: FloatArray = Pytree.field()
    target_rgb_mean_variance_for_object_evolution: FloatArray = Pytree.field()
    alpha_multiplier_for_evolved_gaussian: FloatArray = Pytree.field()
    crp_alpha_for_new_background_gaussian_in_step_model: FloatArray = Pytree.field()

    xyz_cov_pcnt_object_initialization: FloatArray = Pytree.field()
    xyz_mean_pcnt_object_initialization: FloatArray = Pytree.field()
    rgb_var_pcnt_object_initialization: FloatArray = Pytree.field()
    target_rgb_mean_variance_object_initialization: FloatArray = Pytree.field()
    n_unobserved_frames_to_object_gaussian_reset: IntArray = Pytree.field()

    ## Inference hyperparams ##
    always_accept_assoc_depth_move: bool = Pytree.static()
    initial_new_gaussian_prior_params: NewGaussianPriorParams[FloatFromDiscreteSet] = (
        Pytree.field()
    )
    repopulate_depth_nonreturns: bool = Pytree.static()
    do_pose_update: bool = Pytree.static()
    infer_background_evolution_params: bool = Pytree.static()

    ## Misc. ##
    running_simulate: bool = Pytree.field()
    # To make sure we don't have too many pixels with exactly the same
    # floating point value (which inference will try to exploit with Gaussians with
    # absurdly tight RGB distributions), some noise is added to the RGB values
    # before processing.  This is the standard deviation of that noise.
    rgb_noisefloor_std: FloatArray = Pytree.field()


## Next steps:
# To finish the refactor, move
# background_rigidity into the new class.
# Then, I can set up a prior on the params.
# Then, I can add inference.

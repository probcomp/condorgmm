from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as Rot
from jax.tree_util import register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp
from jax.scipy.special import i1e
import genjax

Array: TypeAlias = jax.Array
Float: TypeAlias = Array
Int: TypeAlias = Array
Quaternion: TypeAlias = Array


def multiply_quats(q1, q2):
    return (Rot.from_quat(q1) * Rot.from_quat(q2)).as_quat()


def multiply_quat_and_vec(q, vs):
    return Rot.from_quat(q).apply(vs)


@register_pytree_node_class
class Pose:
    def __init__(self, position, quaternion):
        self._position = position
        self._quaternion = quaternion

    identity_quaternion = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)

    @property
    def unit_quaternion(self):
        raise Warning(
            "Use `identity_quaternion` instead, a unit quaternion is any quat with norm 1!"
        )
        return self.identity_quaternion

    @property
    def pos(self):
        return self._position

    position = pos

    @property
    def xyzw(self):
        return self._quaternion

    quat = xyzw
    quaternion = xyzw

    @property
    def wxyz(self):
        return jnp.concatenate(
            [self.quaternion[..., 3:4], self.quaternion[..., :3]], axis=-1
        )

    @property
    def posquat(self):
        return jnp.concatenate([self.pos, self.quat], axis=-1)

    @property
    def rot(self):
        return Rot.from_quat(self.xyzw)

    def normalize(self):
        quat = self.quat / jnp.linalg.norm(self.quat, axis=-1, keepdims=True)
        return Pose(self.pos, quat)

    def flatten(self):
        return self.pos, self.xyzw

    def tree_flatten(self):
        return ((self.pos, self.xyzw), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def copy(self):
        return Pose(jnp.array(self.pos), jnp.array(self.quat))

    @property
    def flat(self):
        return self.flatten()

    @property
    def shape(self):
        return self.pos.shape[:-1]

    def reshape(self, *args):
        shape = jax.tree.leaves(args)
        return Pose(self.pos.reshape(shape + [3]), self.quat.reshape(shape + [4]))

    def __len__(self):
        return self.pos.shape[0]

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        self.current += 1
        if self.current <= len(self):
            return self[self.current - 1]
        raise StopIteration

    def __getitem__(self, index):
        return Pose(self.pos[index], self.quat[index])

    def slice(self, i):
        return Pose(self.pos[i], self.quat[i])

    def as_matrix(self):
        pose_matrix = jnp.zeros((*self.pos.shape[:-1], 4, 4))
        pose_matrix = pose_matrix.at[..., :3, :3].set(
            Rot.from_quat(self.xyzw).as_matrix()
        )
        pose_matrix = pose_matrix.at[..., :3, 3].set(self.pos)
        pose_matrix = pose_matrix.at[..., 3, 3].set(1.0)
        return pose_matrix

    @staticmethod
    def identity():
        return Pose(
            jnp.zeros(3, dtype=jnp.float32),
            jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32),
        )

    eye = identity
    id = identity

    def apply(self, vec: Array) -> Array:
        return Rot.from_quat(self.xyzw).apply(vec) + self.pos

    transform_points = apply

    def __call__(self, vec: Array) -> Array:
        return self.apply(vec)

    def compose(self, pose: "Pose") -> "Pose":
        return Pose(self.apply(pose.pos), multiply_quats(self.xyzw, pose.xyzw))

    def __add__(self, pose: "Pose") -> "Pose":
        # NOTE: this is useful for gradient updates.
        return Pose(self.pos + pose.pos, self.quat + pose.quat)

    def __sub__(self, pose: "Pose") -> "Pose":
        # NOTE: this is useful for gradient updates.
        return Pose(self.pos - pose.pos, self.quat - pose.quat)

    def scale(self, scale: Float) -> "Pose":
        # NOTE: this is useful for gradient updates.
        return Pose(self.pos * scale, self.quat * scale)

    def __mul__(self, scale: Float) -> "Pose":
        return self.scale(scale)

    def __matmul__(self, pose: "Pose") -> "Pose":
        # TODO: Add test, in particular to lock in matmul vs mul.
        return self.compose(pose)

    @staticmethod
    def concatenate_poses(pose_list):
        return Pose(
            jnp.concatenate([pose.pos for pose in pose_list], axis=0),
            jnp.concatenate([pose.quat for pose in pose_list], axis=0),
        )

    def concat(self, poses, axis=0):
        return Pose(
            jnp.concatenate([self.pos, poses.pos], axis=axis),
            jnp.concatenate([self.quat, poses.quat], axis=axis),
        )

    @staticmethod
    def stack_poses(pose_list):
        return Pose(
            jnp.stack([pose.pos for pose in pose_list]),
            jnp.stack([pose.quat for pose in pose_list]),
        )

    def split(self, n):
        return [
            Pose(ps, qs)
            for (ps, qs) in zip(
                jnp.array_split(self.pos, n), jnp.array_split(self.quat, n)
            )
        ]

    def inv(self):
        R_inv = Rot.from_quat(self.xyzw).inv()
        return Pose(-R_inv.apply(self.pos), R_inv.as_quat())

    inverse = inv

    def __str__(self):
        return f"Pose(position={repr(self.pos)}, quaternion={repr(self.xyzw)})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_matrix(matrix):
        return Pose(matrix[..., :3, 3], Rot.from_matrix(matrix[..., :3, :3]).as_quat())

    @staticmethod
    def from_xyzw(xyzw):
        return Pose(jnp.zeros((*xyzw.shape[:-1], 1)), xyzw)

    from_quat = from_xyzw

    @staticmethod
    def from_pos(position_vec):
        return Pose(
            position_vec,
            jnp.tile(
                jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32),
                (*position_vec.shape[:-1], 1),
            ),
        )

    from_translation = from_pos

    @staticmethod
    def from_vec(posxyzw):
        return Pose(posxyzw[:3], posxyzw[3:])

    @staticmethod
    def from_pos_matrix(pos, matrix):
        return Pose(pos[..., :3], Rot.from_matrix(matrix[..., :3, :3]).as_quat())


### Pose utilities ###
def camera_from_position_and_target(
    position,
    target=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
    up=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
):
    z = target - position
    z = z / jnp.linalg.norm(z)

    x = jnp.cross(z, up)
    x = x / jnp.linalg.norm(x)

    y = jnp.cross(z, x)
    y = y / jnp.linalg.norm(y)

    rotation_matrix = jnp.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])
    return Pose(position, Rot.from_matrix(rotation_matrix).as_quat())


### Rotation distributions ###
@genjax.Pytree.dataclass
class UniformRot(genjax.ExactDensity):
    def logpdf(self, quat):
        # q and -q both have pdf 1/(2π^2) in S^3, so
        # the pdf of q in SO(3) is 1/π^2.
        return -jnp.log(jnp.pi**2)

    def sample(self, key):
        v = jax.random.normal(key, (4,))
        return v / jnp.linalg.norm(v)


uniform_rot = UniformRot()


def vmf_normalizing_constant(conc):
    return jnp.log(conc) - (
        jnp.log(4) + 2 * jnp.log(jnp.pi) + jnp.log(i1e(conc)) + conc
    )


@genjax.Pytree.dataclass
class VMFOnRot(genjax.ExactDensity):
    def logpdf(self, obs, mean, conc):
        return vmf_normalizing_constant(conc) + jax.nn.logsumexp(
            jnp.array(
                [(conc * jnp.dot(obs, mean)), (conc * jnp.dot(-obs, mean))],
                dtype=jnp.float32,
            )
        )

    def sample(self, key, mean, conc):
        return tfp.distributions.VonMisesFisher(mean, conc).sample(seed=key)


vmf_on_rot = VMFOnRot()


### Pose distributions ###
@genjax.Pytree.dataclass
class UniformPose(genjax.ExactDensity):
    def logpdf(self, pose, low, high):
        position = pose.pos
        valid = (low <= position) & (position <= high)
        position_scores = jnp.log(
            (valid * 1.0) * (jnp.ones_like(position) / (high - low))
        )
        return position_scores.sum() + uniform_rot.logpdf(pose.quat)

    def sample(self, key, low, high):
        keys = jax.random.split(key, 2)
        pos = jax.random.uniform(keys[0], (3,)) * (high - low) + low
        quat = uniform_rot()(keys[1])
        return Pose(pos, quat)


uniform_pose = UniformPose()


@genjax.Pytree.dataclass
class GaussianVMF(genjax.ExactDensity):
    def logpdf(self, pose, mean_pose, std, conc):
        translation_score = tfp.distributions.MultivariateNormalDiag(
            mean_pose.pos, jnp.ones(3) * std
        ).log_prob(pose.pos)
        rotation_score = vmf_on_rot.logpdf(pose.quat, mean_pose.quat, conc)
        return translation_score + rotation_score

    def sample(self, key, mean_pose, std, conc):
        k1, k2 = jax.random.split(key)
        x = jax.random.multivariate_normal(k1, mean_pose.pos, std**2 * jnp.eye(3))
        q = vmf_on_rot(mean_pose.quat, conc)(k2)
        return Pose(x, q)


gaussian_vmf = GaussianVMF()

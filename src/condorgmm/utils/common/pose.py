import numpy as np
from scipy.spatial.transform import Rotation as Rot
from jax.scipy.spatial.transform import Rotation as RotJax
import jax.numpy as jnp
import warp as wp


def _jax_numpy_toggle(val):
    if isinstance(val, Pose):
        val = val.posquat
    if isinstance(val, jnp.ndarray):
        return (jnp, RotJax)
    return (np, Rot)


class Pose:
    def __init__(self, posquat):
        if isinstance(posquat, Pose):
            self._posquat = posquat.posquat
        else:
            self._posquat = posquat
        assert self._posquat.shape[-1] == 7

    @staticmethod
    def from_pos_and_quat(pos, quat):
        _np, _ = _jax_numpy_toggle(pos)
        return Pose(_np.concatenate([pos, quat], axis=-1))

    @staticmethod
    def from_translation(translation):
        _np, _ = _jax_numpy_toggle(translation)
        return Pose(
            _np.concatenate([translation, _np.array([0.0, 0.0, 0.0, 1.0])], axis=-1)
        )

    @staticmethod
    def from_matrix(pose_matrix):
        _, _Rot = _jax_numpy_toggle(pose_matrix)
        pos = pose_matrix[..., :3, 3]
        quat = _Rot.from_matrix(pose_matrix[..., :3, :3]).as_quat()
        return Pose.from_pos_and_quat(pos, quat)
    
    def from_rotvec(rotvec):
        _np, _Rot = _jax_numpy_toggle(rotvec)
        return Pose.from_pos_and_quat(
            _np.zeros(3),
            _Rot.from_rotvec(rotvec).as_quat()
        )

    @property
    def posquat(self):
        return self._posquat

    @property
    def pos(self):
        return self._posquat[..., :3]

    @property
    def xyzw(self):
        return self._posquat[..., 3:]

    @property
    def wxyz(self):
        _np, _ = _jax_numpy_toggle(self)
        return _np.concatenate(
            [
                self.quat[..., 6],
                self.quat[..., 3],
                self.quat[..., 4],
                self.quat[..., 5],
            ],
            axis=-1,
        )

    @staticmethod
    def identity():
        return Pose(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    def apply(self, vec):
        _, _Rot = _jax_numpy_toggle(self)
        prev_shape = vec.shape
        vec = vec.reshape(-1, 3)
        transformed_vec = _Rot.from_quat(self.xyzw).apply(vec) + self.pos
        return transformed_vec.reshape(prev_shape)

    transform_points = apply

    def as_matrix(self):
        _np, _Rot = _jax_numpy_toggle(self)

        pose_matrix = _np.zeros((*self.pos.shape[:-1], 4, 4))

        if isinstance(self.pos, jnp.ndarray):
            pose_matrix = pose_matrix.at[..., :3, :3].set(
                _Rot.from_quat(self.xyzw).as_matrix()
            )
            pose_matrix = pose_matrix.at[..., :3, 3].set(self.pos)
            pose_matrix = pose_matrix.at[..., 3, 3].set(1.0)
        else:
            pose_matrix[..., :3, :3] = _Rot.from_quat(self.xyzw).as_matrix()
            pose_matrix[..., :3, 3] = self.pos
            pose_matrix[..., 3, 3] = 1.0
        return pose_matrix

    @property
    def scipy_rotation(self):
        _, _Rot = _jax_numpy_toggle(self)
        return _Rot.from_quat(self.xyzw)

    def inverse(self):
        _, _Rot = _jax_numpy_toggle(self)
        R_inv = _Rot.from_quat(self.xyzw).inv()
        return Pose.from_pos_and_quat(-R_inv.apply(self.pos), R_inv.as_quat())

    inv = inverse

    def compose(self, pose: "Pose") -> "Pose":
        slf, pose = cohere_jax_numpy(self, pose)

        return Pose.from_pos_and_quat(
            slf.apply(pose.pos), (slf.scipy_rotation * pose.scipy_rotation).as_quat()
        )

    def __matmul__(self, pose: "Pose") -> "Pose":
        # TODO: Add test, in particular to lock in matmul vs mul.
        return self.compose(pose)

    # Multidimensional Pose
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
        return Pose(self.posquat[index])

    def slice(self, i):
        return Pose(self.posquat[i])
    
    def to_wp_transform(self):
        assert self.posquat.shape == (7,)
        return wp.transform(self.posquat[:3], self.posquat[3:])


def cohere_jax_numpy(a: Pose, b: Pose):
    if isinstance(a.posquat, jnp.ndarray) or isinstance(b.posquat, jnp.ndarray):
        return Pose(jnp.array(a.posquat)), Pose(jnp.array(b.posquat))
    return a, b

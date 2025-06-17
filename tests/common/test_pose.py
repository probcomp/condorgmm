from condorgmm.utils.common import Pose
import unittest
import numpy as np


class PoseTests(unittest.TestCase):
    def test_pose_properties(self):
        posquat = np.random.rand(7)
        pose = Pose(posquat)

        self.assertTrue(np.allclose(pose.posquat, posquat))
        self.assertTrue(np.allclose(pose.pos, posquat[:3]))
        self.assertTrue(np.allclose(pose.xyzw, posquat[3:]))

    def test_inverse(self):
        posquat = np.random.rand(7)
        pose = Pose(posquat)
        inv_pose = pose.inv()

        matrix = pose.as_matrix()
        inverse_matrix = inv_pose.as_matrix()

        assert np.allclose(
            np.linalg.inv(matrix), inverse_matrix, atol=1e-4
        ), f"{np.abs(np.linalg.inv(matrix) - inverse_matrix).max()}"

    def test_compose(self):
        posquat1 = np.random.rand(7)
        posquat2 = np.random.rand(7)
        pose1 = Pose(posquat1)
        pose2 = Pose(posquat2)
        composed_pose = pose1 @ pose2

        matrix1 = pose1.as_matrix()
        matrix2 = pose2.as_matrix()
        composed_matrix = matrix1 @ matrix2

        assert np.allclose(
            composed_pose.as_matrix(), composed_matrix, atol=1e-4
        ), f"{np.abs(composed_pose.as_matrix() - composed_matrix).max()}"

    def test_transform_points(self):
        posquat = np.random.rand(7)
        pose = Pose(posquat)
        points = np.random.rand(100, 3)

        transformed_points = pose.transform_points(points)

        matrix = pose.as_matrix()
        homogenous_points = np.concatenate(
            [points, np.ones((points.shape[0], 1))], axis=-1
        )
        transformed_points_2 = (homogenous_points @ matrix.T)[..., :3]

        assert np.allclose(
            transformed_points, transformed_points_2, atol=1e-4
        ), f"{np.abs(transformed_points - transformed_points_2).max()}"

    def test_iterate(self):
        posquats = np.random.rand(100, 7)
        p = Pose(posquats)
        iterations = 0
        for i, subpose in enumerate(p):
            iterations += 1
            assert subpose.posquat.shape == (7,)
            assert np.allclose(subpose.posquat, posquats[i])

        assert iterations == 100

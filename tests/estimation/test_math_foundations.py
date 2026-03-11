from __future__ import annotations

import unittest

import numpy as np

from sim.estimation.core import (
    apply_minimum_variance,
    regularize_covariance,
    symmetrize_covariance,
)
from sim.estimation.math import (
    finite_difference_jacobian,
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_vector_to_quaternion,
    skew_symmetric,
)


class QuaternionMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(1234)

    def test_quaternion_normalization_preserves_unit_norm(self) -> None:
        for _ in range(50):
            quaternion = self.rng.normal(size=4)
            normalized = normalize_quaternion(quaternion)
            self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0, places=12)

    def test_zero_quaternion_normalizes_to_identity(self) -> None:
        normalized = normalize_quaternion(np.zeros(4, dtype=float))
        np.testing.assert_allclose(normalized, np.array([1.0, 0.0, 0.0, 0.0], dtype=float))

    def test_quaternion_inverse_recovers_identity(self) -> None:
        for _ in range(50):
            quaternion = normalize_quaternion(self.rng.normal(size=4))
            recovered = quaternion_multiply(quaternion, quaternion_inverse(quaternion))
            np.testing.assert_allclose(
                recovered,
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                atol=1e-12,
            )

    def test_rotation_vector_exponential_matches_first_order_near_zero(self) -> None:
        rotation_vector = np.array([1.0e-9, -2.0e-9, 3.0e-9], dtype=float)
        exact = rotation_vector_to_quaternion(rotation_vector)
        first_order = normalize_quaternion(
            np.array([1.0, *(0.5 * rotation_vector)], dtype=float)
        )
        np.testing.assert_allclose(exact, first_order, atol=1e-15, rtol=1e-12)

    def test_rotation_matrices_remain_orthonormal(self) -> None:
        for _ in range(50):
            rotation_vector = self.rng.normal(size=3)
            rotation_matrix = quaternion_to_rotation_matrix(
                rotation_vector_to_quaternion(rotation_vector)
            )
            np.testing.assert_allclose(
                rotation_matrix.T @ rotation_matrix,
                np.eye(3, dtype=float),
                atol=1e-12,
            )
            self.assertAlmostEqual(float(np.linalg.det(rotation_matrix)), 1.0, places=12)


class JacobianTests(unittest.TestCase):
    def test_rotation_vector_to_quaternion_jacobian_near_zero_matches_first_order(self) -> None:
        point = np.zeros(3, dtype=float)
        numeric = finite_difference_jacobian(rotation_vector_to_quaternion, point)
        analytic = np.vstack(
            [
                np.zeros((1, 3), dtype=float),
                0.5 * np.eye(3, dtype=float),
            ]
        )
        np.testing.assert_allclose(numeric, analytic, atol=1e-9, rtol=1e-6)

    def test_skew_symmetric_jacobian_matches_finite_difference(self) -> None:
        point = np.array([0.3, -0.5, 0.8], dtype=float)
        numeric = finite_difference_jacobian(lambda vector: skew_symmetric(vector).reshape(-1), point)

        basis_x = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ).reshape(-1)
        basis_y = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=float,
        ).reshape(-1)
        basis_z = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ).reshape(-1)
        analytic = np.column_stack([basis_x, basis_y, basis_z])
        np.testing.assert_allclose(numeric, analytic, atol=1e-9, rtol=1e-7)


class CovarianceUtilityTests(unittest.TestCase):
    def test_symmetrize_covariance_averages_matrix_with_transpose(self) -> None:
        covariance = np.array(
            [
                [2.0, 3.0, -1.0],
                [1.0, 4.0, 0.5],
                [2.0, -0.5, 1.0],
            ],
            dtype=float,
        )
        symmetrized = symmetrize_covariance(covariance)
        np.testing.assert_allclose(symmetrized, 0.5 * (covariance + covariance.T))

    def test_apply_minimum_variance_clamps_diagonal(self) -> None:
        covariance = np.array(
            [
                [1.0e-12, 0.1],
                [0.1, -2.0],
            ],
            dtype=float,
        )
        regularized = apply_minimum_variance(covariance, min_variance=1.0e-6)
        np.testing.assert_allclose(
            np.diag(regularized),
            np.array([1.0e-6, 1.0e-6], dtype=float),
        )

    def test_regularize_covariance_symmetrizes_and_clamps(self) -> None:
        covariance = np.array(
            [
                [1.0e-12, 4.0],
                [1.0, -5.0],
            ],
            dtype=float,
        )
        regularized = regularize_covariance(covariance, min_variance=1.0e-5)
        np.testing.assert_allclose(regularized, regularized.T)
        np.testing.assert_allclose(
            np.diag(regularized),
            np.array([1.0e-5, 1.0e-5], dtype=float),
        )


if __name__ == "__main__":
    unittest.main()

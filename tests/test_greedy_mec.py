import unittest

import numpy as np

from mec import greedy_mec

from utilities import generate_distribution


class GreedyApproximationTest(unittest.TestCase):
    def test_runs(self):
        for uniform in [True, False]:
            x = generate_distribution(10, uniform)
            y = generate_distribution(20, uniform)
            z = generate_distribution(5, uniform)
            greedy_mec(x, y, z)
            greedy_mec(x, y, z, sparse=True)

    def test_dense_marginalizes(self):
        for uniform in [True, False]:
            x = generate_distribution(10, uniform)
            y = generate_distribution(20, uniform)
            z = generate_distribution(5, uniform)
            coupling = greedy_mec(x, y, z)
            self.assertTrue(np.allclose(np.sum(coupling), 1))
            self.assertTrue(np.allclose(np.sum(coupling, axis=(1, 2)), x))
            self.assertTrue(np.allclose(np.sum(coupling, axis=(0, 2)), y))
            self.assertTrue(np.allclose(np.sum(coupling, axis=(0, 1)), z))

    def test_sparse_marginalizes(self):
        for uniform in [True, False]:
            x = generate_distribution(10, uniform)
            y = generate_distribution(20, uniform)
            z = generate_distribution(5, uniform)
            coupling = greedy_mec(x, y, z, sparse=True)
            self.assertTrue(np.allclose(sum(coupling.values()), 1))
            reconstructed_marginals = [
                np.zeros_like(m, dtype=np.float128) for m in [x, y, z]
            ]
            for idx, mass in coupling.items():
                for i, j in enumerate(idx):
                    reconstructed_marginals[i][j] += mass
            for reconstructed_marginal, marginal in zip(
                reconstructed_marginals, [x, y, z]
            ):
                self.assertTrue(np.isclose(reconstructed_marginal, marginal).all())

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            greedy_mec([0.5, 0.5], "invalid")
        with self.assertRaises(TypeError):
            greedy_mec("invalid", [0.5, 0.5])
        with self.assertRaises(TypeError):
            greedy_mec([0.5, 0.5], [0.5, 0.5], sparse="invalid")

    def test_non_1d_array(self):
        with self.assertRaises(ValueError):
            greedy_mec(np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([0.5, 0.5]))
        with self.assertRaises(ValueError):
            greedy_mec(
                np.array([0.5, 0.5]), np.array([[0.5, 0.5], [0.5, 0.5]]), sparse=True
            )

    def test_invalid_distribution(self):
        with self.assertRaises(ValueError):
            greedy_mec(np.array([0.5, 0.5]), np.array([0.5, 0.6]))
        with self.assertRaises(ValueError):
            greedy_mec(np.array([0.5, 0.5, 0.1]), np.array([0.5, 0.5]), sparse=True)


if __name__ == "__main__":
    unittest.main()

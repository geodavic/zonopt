from zonopt.polytope import Polytope, Zonotope, Hyperplane, Halfspace, Cube
from zonopt.polytope.zonotope import express_as_subset_sum
from zonopt.polytope.exceptions import GeometryError
import numpy as np
import torch
import unittest


class TestZonotope(unittest.TestCase):
    def setUp(self):
        self.C = Cube(3, as_zonotope=True)
        self.C_t = Cube(3, as_zonotope=True, use_torch=True)

    def test_change_generators(self):
        delta = np.array([1, 1, 1])
        C = self.C.copy()
        C.generators += delta

        np.testing.assert_almost_equal(
            C.generators, np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        )
        np.testing.assert_almost_equal(C.vertices[-1], np.array([4, 4, 4]))

    def test_change_translation(self):
        delta = np.array([1, 1, 1])
        C = self.C.copy()
        C.translation += delta

        np.testing.assert_almost_equal(C.translation, delta)
        np.testing.assert_almost_equal(C.vertices[-1], np.array([2, 2, 2]))

    def test_change_order_doesnt_matter(self):
        delta_t = np.array([1, 1, 1])
        C = self.C.copy()
        D = self.C.copy()

        C.generators *= 2
        C.translation += delta_t

        D.translation += delta_t
        D.generators *= 2

        np.testing.assert_almost_equal(C.vertices, D.vertices)

    def test_torch_change_generators(self):
        delta = torch.tensor([1, 1, 1], dtype=torch.float64)
        C = self.C_t.copy()
        C.generators += delta

        torch.testing.assert_close(
            C.generators,
            torch.tensor([[2, 1, 1], [1, 2, 1], [1, 1, 2.0]], dtype=torch.float64),
        )
        np.testing.assert_almost_equal(C.vertices[-1], np.array([4, 4, 4]))

    def test_torch_change_translation(self):
        delta = torch.tensor([1, 1, 1], dtype=torch.float64)
        C = self.C_t.copy()
        C.translation += delta

        torch.testing.assert_close(C.translation, delta)
        np.testing.assert_almost_equal(C.vertices[-1], np.array([2, 2, 2]))

    def test_express_as_subset(self):
        Z = Zonotope.random(5, 3)
        for _ in range(10):
            subset = np.random.randint(0, 2, 5)
            target = sum([g for i, g in enumerate(Z.generators) if subset[i]])
            if isinstance(target, int):
                target = np.zeros(3)
            is_sum, coeffs = express_as_subset_sum(target, Z)
            self.assertTrue(is_sum)
            np.testing.assert_equal(subset, np.array(coeffs))

    def test_zonotope_vertices_are_subset_sums(self):
        Z = Zonotope.random(5, 3)
        for v in Z.vertices:
            is_sum, _ = express_as_subset_sum(v, Z)
            self.assertTrue(is_sum)

    def test_lift(self):
        for dimension, rank in [(2, 4), (3, 5), (4, 6)]:
            Z = Zonotope.random(rank, dimension, seed=1, random_translation=True)
            # Vertices lift to 0/1 vectors that match the subset sum
            for v in Z.vertices:
                epsilon = Z.lift(v)
                _, coeffs = express_as_subset_sum(v, Z)
                np.testing.assert_equal(epsilon, np.array(coeffs, dtype=np.float64))
            # Midpoint of an edge lifts to a point with a single fractional coordinate
            v0 = Z.vertices[0]
            e0 = Z.lift(v0)
            for i in range(rank):
                e1 = e0.copy()
                e1[i] = 1 - e1[i]
                v1 = Z.generators.T @ e1 + Z.translation
                if Z.has_vertex(v1):
                    epsilon = Z.lift((v0 + v1) / 2)
                    np.testing.assert_almost_equal(epsilon, (e0 + e1) / 2)
                    break
            # Points outside raise
            with self.assertRaises(GeometryError):
                Z.lift(Z.vertices[0] + 10 * np.ones(dimension))

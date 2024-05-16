from zonopt.polytope import Polytope, Zonotope, Hyperplane, Halfspace, Cube
from zonopt.polytope.zonotope import express_as_subset_sum
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
        for _ in range(5):
            subset = np.random.randint(0, 2, 5)
            target = sum([g for i, g in enumerate(Z.generators) if subset[i]])
            if isinstance(target, int):
                target = np.zeros(5)
            is_sum, coeffs = express_as_subset_sum(
                target, [g for g in Z.generators], np.zeros(3)
            )
            self.assertTrue(is_sum)
            np.testing.assert_equal(subset, np.array(coeffs))

    def test_zonotope_vertices_are_subset_sums(self):
        Z = Zonotope.random(5, 3)
        for v in Z.vertices:
            is_sum, _ = express_as_subset_sum(v, [g for g in Z.generators], np.zeros(3))
            self.assertTrue(is_sum)

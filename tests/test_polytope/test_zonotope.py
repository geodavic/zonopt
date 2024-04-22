from zonopt.polytope import Polytope, Hyperplane, Halfspace, Cube
import numpy as np
import unittest


class TestZonotope(unittest.TestCase):
    def setUp(self):
        self.C1 = Cube(3, as_zonotope=True)
        self.C2 = Cube(3, as_zonotope=True)
        self.C3 = Cube(3, as_zonotope=True)
        self.C4 = Cube(3, as_zonotope=True)

    def test_change_generators(self):
        delta = np.array([1, 1, 1])
        self.C1.generators += delta

        np.testing.assert_almost_equal(
            self.C1.generators, np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        )
        np.testing.assert_almost_equal(self.C1.vertices[-1], np.array([4, 4, 4]))

    def test_change_translation(self):
        delta = np.array([1, 1, 1])
        self.C2.translation += delta

        np.testing.assert_almost_equal(self.C2.translation, delta)
        np.testing.assert_almost_equal(self.C2.vertices[-1], np.array([2, 2, 2]))

    def test_change_order_doesnt_matter(self):
        delta_t = np.array([1, 1, 1])

        self.C3.generators *= 2
        self.C3.translation += delta_t

        self.C4.translation += delta_t
        self.C4.generators *= 2

        np.testing.assert_almost_equal(self.C3.vertices, self.C4.vertices)

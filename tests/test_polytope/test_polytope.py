from zonopt.polytope import Polytope, Hyperplane, Halfspace
from zonopt.metrics.hausdorff import _distance_to_polytope_l2
import numpy as np
import unittest


class TestPolytope(unittest.TestCase):
    def setUp(self):
        self.hyp = Hyperplane([1, 1], 1)
        self.half = Halfspace([1, 1], 1)
        self.P1 = Polytope(points=[[1, 0], [0, 1], [-1, 0], [0, -1]])
        self.P2 = Polytope(points=[[0, 0], [2, 0], [1, 2]])
        self.P3 = Polytope(
            points=[
                [0, 0, 0],
                [1, 2, 3],
                [0, 1, 1],
                [-1, 2, 0],
                [1, 3, 4],
                [0, 4, 3],
                [-1, 3, 1],
                [0, 5, 4],
            ]
        )

    def test_mul(self):
        P = 2 * self.P1
        H = Halfspace([1, 1], 2)
        self.assertTrue(H in P.halfspaces)

    def test_halfspaces(self):
        self.assertTrue(self.half in self.P1.halfspaces)

    def test_boundary(self):
        self.assertTrue(self.half.boundary == self.hyp)

    def test_contains(self):
        x = [9, -8]
        y = [-5, -5]
        z = [1, 0.5]

        self.assertTrue(self.hyp.contains(x))
        self.assertFalse(self.hyp.contains(y))
        self.assertTrue(self.half.contains(y))
        self.assertTrue(self.P2.contains(z))

    def test_dimension(self):
        self.assertEqual(self.P1.dimension, 2)
        self.assertEqual(self.P3.dimension, 3)

    def test_centrally_symmetric(self):
        self.assertTrue(self.P1.is_centrally_symmetric())

    def test_barycenter(self):
        b1 = np.array([0, 0])
        b2 = np.array([1, 2 / 3])
        b3 = np.array([0, 20 / 8, 16 / 8])

        np.testing.assert_almost_equal(b1, self.P1.barycenter)
        np.testing.assert_almost_equal(b2, self.P2.barycenter)
        np.testing.assert_almost_equal(b3, self.P3.barycenter)

    def test_face_dimension(self):
        self.assertEqual(self.P1.face_dimension([1, 1]), -1)
        self.assertEqual(self.P1.face_dimension([1, 0]), 0)
        self.assertEqual(self.P1.face_dimension([0.5, 0.5]), 1)

        _, proj, _ = _distance_to_polytope_l2([-3, 4, 4], self.P3)
        self.assertEqual(self.P3.face_dimension(proj), 1)

        # TODO figure out how to select a face of given dimension
        # and test that

from zonopt.metrics import (
    distance_to_polytope,
    distance_to_hyperplane,
    hausdorff_distance,
)
from zonopt.polytope import UnitBall, Polytope, Cube, Zonotope
from zonopt.metrics.hausdorff import _distance_to_zonotope_l2
import scipy as sp
from qpsolvers import solve_qp
from zonopt.config import global_config
import numpy as np
import unittest


class TestDistanceToPolytopeL2(unittest.TestCase):
    def setUp(self):
        self.metric = 2
        self.diamond = UnitBall(2, 1)
        self.cube = UnitBall(2, np.infty)
        scale = 1 / np.sqrt(2)
        self.octagon = Polytope(
            points=np.concatenate([(scale * self.cube).vertices, self.diamond.vertices])
        )

    def distance_to_polytope(self, x, P):
        return distance_to_polytope(x, P, metric=2)

    def test_diamond(self):
        x = np.array([-1, 1])
        y = np.array([0, 0])

        dist, proj, _ = self.distance_to_polytope(x, self.diamond)
        self.assertAlmostEqual(dist, np.sqrt(2) / 2)
        self.assertTrue(np.allclose(proj, x / 2))

        dist, proj, _ = self.distance_to_polytope(y, self.diamond)
        self.assertAlmostEqual(dist, 0)
        self.assertTrue(np.allclose(proj, y))

    def test_cube(self):
        x = np.array([-2, 2])
        y = np.array([0.5, 0.5])

        dist, proj, _ = self.distance_to_polytope(x, self.cube)
        self.assertAlmostEqual(dist, np.sqrt(2))
        self.assertTrue(np.allclose(proj, x / 2))

        dist, proj, _ = self.distance_to_polytope(y, self.cube)
        self.assertAlmostEqual(dist, 0)
        self.assertTrue(np.allclose(proj, y))

    def test_octagon(self):
        x = np.array([-2, 1])
        y = np.array([0.25, 0.25])

        dist, proj, _ = self.distance_to_polytope(x, self.octagon)
        supp = self.octagon.supporting_halfspaces(proj)
        self.assertEqual(len(supp), 1)
        dist2 = distance_to_hyperplane(x, supp[0].boundary)
        self.assertAlmostEqual(dist, dist2)

        dist, proj, _ = self.distance_to_polytope(y, self.octagon)
        self.assertAlmostEqual(dist, 0)
        self.assertTrue(np.allclose(proj, y))


class TestHausdorffDistance(unittest.TestCase):
    """
    TODO: this could use some more cases.
    """

    def setUp(self):
        self.cube = UnitBall(2, np.infty)

        rot_angle = np.pi / 4
        rot = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle)],
                [np.sin(rot_angle), np.cos(rot_angle)],
            ]
        )
        self.cube_rotated = Polytope(points=[rot @ p for p in self.cube.vertices])

        self.thresh = 0.99999999

    def test_multiplicity(self):
        dist, ps, qs = hausdorff_distance(
            self.cube, self.cube_rotated, threshold=self.thresh
        )

        self.assertAlmostEqual(dist, np.sqrt(2) - 1)
        self.assertEqual(len(ps), 8)
        self.assertEqual(len(qs), 8)


class TestZonotopeDistance(unittest.TestCase):
    def vertex_qp(self, x, P):
        """The original vertex-based quadratic program, kept here as a reference."""
        V = P.vertices
        Q = sp.linalg.block_diag(np.eye(len(x)), np.zeros((len(V), len(V))))
        q = -np.concatenate([x, np.zeros(len(V))])
        A = np.concatenate([np.eye(len(x)), -V.T], axis=1)
        A = np.concatenate([A, [[0] * len(x) + [1] * len(V)]], axis=0)
        b = np.array([0] * len(x) + [1], np.float64)
        lb = np.array([-np.inf] * len(x) + [0] * len(V), np.float64)
        sol = solve_qp(Q, q, A=A, b=b, lb=lb, solver="osqp", polish=1, eps_abs=1e-11, eps_rel=1e-11, max_iter=4000000)
        return np.linalg.norm(sol[: len(x)] - x), sol[: len(x)]

    def test_cube_coordinates_match_vertex_qp(self):
        for dimension, rank in [(2, 4), (3, 6), (4, 6)]:
            Z = Zonotope.random(rank, dimension, seed=1, positive=False, random_translation=True)
            P = Polytope.random(8, dimension, seed=5)
            for x in P.vertices:
                d_qp, proj_qp = self.vertex_qp(x, Z)
                d_ls, proj_ls, epsilon = _distance_to_zonotope_l2(x, Z)
                self.assertAlmostEqual(d_qp, d_ls, places=7)
                np.testing.assert_almost_equal(proj_qp, proj_ls, decimal=6)
                np.testing.assert_almost_equal(
                    Z.generators.T @ epsilon + Z.translation, proj_ls
                )
                self.assertTrue(np.all(epsilon >= 0) and np.all(epsilon <= 1))

    def test_interior_point(self):
        Z = Zonotope.random(5, 3, seed=1)
        x = Z.barycenter
        d, proj, _ = distance_to_polytope(x, Z)
        self.assertEqual(d, 0)
        np.testing.assert_almost_equal(proj, x)

    def test_hausdorff_pruning_matches_brute_force(self):
        for dimension, rank, seed in [(2, 5, 0), (3, 6, 1), (3, 8, 2), (4, 6, 3)]:
            P = Polytope.random(10, dimension, seed=seed)
            Z = Zonotope.random(rank, dimension, seed=seed, positive=False, random_translation=True)
            dist, pp, qq = hausdorff_distance(P, Z)
            brute = max(
                max(distance_to_polytope(p, Z)[0] for p in P.vertices),
                max(distance_to_polytope(q, P)[0] for q in Z.vertices),
            )
            self.assertAlmostEqual(dist, brute, places=9)
            self.assertAlmostEqual(np.linalg.norm(pp[0] - qq[0]), dist, places=9)
            self.assertTrue(P.has_vertex(pp[0]) or Z.has_vertex(qq[0]))

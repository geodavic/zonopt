from zonopt.polytope import Polytope, Zonotope, Hyperplane, Halfspace, Cube
from zonopt.polytope.zonotope import express_as_subset_sum, zonotope_geometry
from zonopt.polytope.utils import hull_from_affine
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

    def test_geometry_matches_brute_force(self):
        """
        Facet enumeration gives the same vertices and facets as the convex hull
        of all 2^n cubical vertices.
        """
        def facet_set(equations):
            return {tuple(np.round(e, 7)) for e in equations}

        def vertex_set(vertices):
            return {tuple(np.round(v, 9)) for v in vertices}

        cases = [(2, 4), (2, 7), (3, 5), (3, 9), (4, 6), (4, 8), (5, 7)]
        for dimension, rank in cases:
            for seed in range(3):
                Z = Zonotope.random(
                    rank, dimension, seed=seed, positive=False, random_translation=True
                )
                hull = hull_from_affine(Z.generators, Z.translation)
                self.assertTrue(Z.is_general_position)
                self.assertEqual(
                    vertex_set(Z.vertices), vertex_set(hull.points[hull.vertices])
                )
                self.assertEqual(
                    facet_set(Z.equations), facet_set(np.unique(hull.equations, axis=0))
                )

        # Degenerate: fourth generator lies in the plane of the first two
        G = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
        Z = Zonotope(G)
        hull = hull_from_affine(G, np.zeros(3))
        self.assertFalse(Z.is_general_position)
        self.assertEqual(vertex_set(Z.vertices), vertex_set(hull.points[hull.vertices]))
        self.assertEqual(
            facet_set(Z.equations), facet_set(np.unique(hull.equations, axis=0))
        )

        # Cube
        self.assertEqual(len(self.C.vertices), 8)
        self.assertEqual(len(self.C.equations), 6)

    def test_vertices_counterclockwise_in_2d(self):
        Z = Zonotope.random(6, 2, seed=4, positive=False)
        V = Z.vertices - Z.barycenter
        angles = np.arctan2(V[:, 1], V[:, 0])
        self.assertTrue(np.all(np.diff(angles) > 0))

    def test_geometry_cache_invalidated_on_update(self):
        Z = Zonotope.random(5, 3, seed=2)
        V0 = Z.vertices.copy()
        Z.translation = Z.translation + 1.0
        np.testing.assert_almost_equal(Z.vertices, V0 + 1.0)
        Z.generators = 2 * Z.generators
        self.assertEqual(len(Z.vertices), len(V0))
        np.testing.assert_almost_equal(
            Z.vertices.max(axis=0) - Z.vertices.min(axis=0),
            2 * (V0.max(axis=0) - V0.min(axis=0)),
        )

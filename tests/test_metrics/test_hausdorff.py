from zonopt.metrics import distance_to_polytope, distance_to_hyperplane, hausdorff_distance
from zonopt.polytope import UnitBall, Polytope, Cube
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
        self.cube = UnitBall(2,np.infty)

        rot_angle = np.pi/4
        rot = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],[np.sin(rot_angle), np.cos(rot_angle)]])
        self.cube_rotated = Polytope(points=[rot@p for p in self.cube.vertices])

        self.thresh = 0.0000001

    def test_multiplicity(self):
        dist, p, q, mult = hausdorff_distance(self.cube,self.cube_rotated, threshold=self.thresh)

        self.assertAlmostEqual(dist, np.sqrt(2)-1)
        self.assertEqual(mult, 8)

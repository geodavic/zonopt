import numpy as np
from zonopt.polytope import Polytope
from zonopt.plot.utils import get_bounds
import unittest


class TestBounds(unittest.TestCase):
    def setUp(self):
        self.P1 = Polytope(points=[[1, 0], [0, 1], [-1, 0], [0, -1]])
        self.P2 = Polytope(points=[[0, 0], [2, 0], [1, 2]])
        self.P3 = Polytope(
            points=[
                [0, 0],
                [1, 2],
                [0, 1],
                [-1, 2],
                [1, 3],
                [0, 4],
                [-1, 3],
                [0, 5],
            ]
        )

    def test_single_bounds1(self):
        target_max_xy = [1, 1]
        target_min_xy = [-1, -1]

        pred_max_xy, pred_min_xy = get_bounds([self.P1])
        self.assertEqual(target_max_xy, pred_max_xy)
        self.assertEqual(target_min_xy, pred_min_xy)

    def test_single_bounds2(self):
        target_max_xy = [2, 2]
        target_min_xy = [0, 0]

        pred_max_xy, pred_min_xy = get_bounds([self.P2])
        self.assertEqual(target_max_xy, pred_max_xy)
        self.assertEqual(target_min_xy, pred_min_xy)

    def test_single_bounds3(self):
        target_max_xy = [1, 5]
        target_min_xy = [-1, 0]

        pred_max_xy, pred_min_xy = get_bounds([self.P3])
        self.assertEqual(target_max_xy, pred_max_xy)
        self.assertEqual(target_min_xy, pred_min_xy)

    def test_all_bounds(self):
        target_max_xy = [2, 5]
        target_min_xy = [-1, -1]

        pred_max_xy, pred_min_xy = get_bounds([self.P1, self.P2, self.P3])
        self.assertEqual(target_max_xy, pred_max_xy)
        self.assertEqual(target_min_xy, pred_min_xy)

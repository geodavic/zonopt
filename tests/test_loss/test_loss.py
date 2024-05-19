from zonopt.polytope import Polytope, Zonotope
from zonopt.polytope.zonotope import express_as_subset_sum
from zonopt.loss.loss import get_facet_normal, get_vertex_on_facet, hausdorff_loss
from zonopt.metrics import hausdorff_distance
import torch
import numpy as np
import unittest


class TestHausdorffLoss(unittest.TestCase):
    def setUp(self):
        # Basic zonotope
        generators = torch.tensor(
            [[1, 0], [1, 1], [0, 1]], requires_grad=True, dtype=torch.float64
        )
        translation = torch.zeros(2, requires_grad=True, dtype=torch.float64)
        self.Z = Zonotope(generators=generators, translation=translation)

        # Rotate cube by pi/4 and pi/6 and move
        cube = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        theta1 = np.pi / 6
        theta2 = np.pi / 4
        rot1 = np.array(
            [[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1), np.cos(theta1)]]
        )
        rot2 = np.array(
            [[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]]
        )
        offset1 = np.array([0, 2])
        offset2 = np.array([1.5, 10])
        cube1 = np.array([rot1 @ v + offset1 for v in cube])
        cube2 = np.array([rot2 @ v + offset2 for v in cube])

        # Will achieve hausdorff distance at type I
        self.P1 = Polytope(points=cube1)
        # Will achieve hausdorff distance at type II
        self.P2 = Polytope(points=cube2)

        # hausdorff distance
        self.haus_dist1, [self.p1], [self.q1] = hausdorff_distance(self.P1, self.Z)
        self.haus_dist2, [self.p2], [self.q2] = hausdorff_distance(self.P2, self.Z)

    def test_typeI_grad(self):
        loss = hausdorff_loss(self.Z, self.P1, self.q1, self.p1)
        loss.backward()

        # Hausdorff loss should be the same as hausdorff distance
        np.testing.assert_almost_equal(loss.item(), self.haus_dist1)

        # p1 should lie on a codimension 1 face.
        halfspaces = self.P1.supporting_halfspaces(self.p1)
        self.assertTrue(len(halfspaces), 1)

        # q1 should be a vertex of Z
        self.assertTrue(self.Z.has_vertex(self.q1))

        eta = halfspaces[0]._a
        _, epsilon = express_as_subset_sum(self.q1, self.Z, return_indices=False)

        for i in range(self.Z.rank):
            for j in range(self.Z.dimension):
                # Algebraic value of d \delta_q / d g_{i,j}
                target_gradient = eta[j] * epsilon[i]
                # Calculated value via loss
                loss_gradient = self.Z.generators.grad[i][j].item()
                np.testing.assert_almost_equal(target_gradient, loss_gradient)

        for i in range(self.Z.dimension):
            # Algebraic value of d \delta_q / d\mu_j
            target_gradient = eta[j]
            # Calculated value via loss
            loss_gradient = self.Z.translation.grad[j].item()
            np.testing.assert_almost_equal(target_gradient, loss_gradient)

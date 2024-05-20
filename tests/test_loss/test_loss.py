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

        # Rotate cube and move
        cube = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        theta1 = np.pi / 6
        theta2 = 0
        rot1 = np.array(
            [[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1), np.cos(theta1)]]
        )
        rot2 = np.array(
            [[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]]
        )
        offset1 = np.array([0, 2])
        offset2 = np.array([-1, 2])
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
        self.assertTrue(len(halfspaces) == 1)

        # q1 should be a vertex of Z
        self.assertTrue(self.Z.has_vertex(self.q1))

        eta = halfspaces[0]._a
        _, epsilon = express_as_subset_sum(self.q1, self.Z, return_indices=False)

        # Check against Proposition 6.12
        for i in range(self.Z.rank):
            for j in range(self.Z.dimension):
                # Algebraic value of d \delta_q / d g_{i,j}
                target_gradient = eta[j] * epsilon[i]
                # Calculated value via loss
                loss_gradient = self.Z.generators.grad[i][j].item()
                np.testing.assert_almost_equal(target_gradient, loss_gradient)

        for j in range(self.Z.dimension):
            # Algebraic value of d \delta_q / d\mu_j
            target_gradient = eta[j]
            # Calculated value via loss
            loss_gradient = self.Z.translation.grad[j].item()
            np.testing.assert_almost_equal(target_gradient, loss_gradient)

    def test_typeII_grad(self):
        loss = hausdorff_loss(self.Z, self.P2, self.q2, self.p2)
        loss.backward()

        # Hausdorff loss should be the same as hausdorff distance
        np.testing.assert_almost_equal(loss.item(), self.haus_dist2)

        # q2 should lie on a codimension 1 face.
        halfspaces = self.Z.supporting_halfspaces(self.q2)
        self.assertTrue(len(halfspaces) == 1)

        # p2 should be a vertex of P2
        self.assertTrue(self.P2.has_vertex(self.p2))

        vertex_on_facet = get_vertex_on_facet(self.Z, halfspaces[0])
        _, epsilon = express_as_subset_sum(
            vertex_on_facet, self.Z, return_indices=False
        )

        # Check against Proposition 6.16
        for i in range(self.Z.rank):
            for j in range(self.Z.dimension):
                Z2 = self.Z.copy(requires_grad=True)
                eta = get_facet_normal(Z2, halfspaces[0])
                target_gradient = -eta[j] * epsilon[i]
                for jp in range(self.Z.dimension):
                    Z2.generators.grad = None
                    etajp = eta[jp]
                    etajp.backward(retain_graph=True)
                    d_etajp = Z2.generators.grad[i][j].item()
                    target_gradient += d_etajp * (self.p2[jp] - vertex_on_facet[jp])

                loss_gradient = self.Z.generators.grad[i][j].item()
                np.testing.assert_almost_equal(loss_gradient, target_gradient.item())

        for j in range(self.Z.dimension):
            eta = get_facet_normal(self.Z, halfspaces[0])
            target_gradient = -eta[j]

            loss_gradient = self.Z.translation.grad[j].item()
            np.testing.assert_almost_equal(loss_gradient, target_gradient.item())

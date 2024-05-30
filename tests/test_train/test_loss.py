from zonopt.polytope import Polytope, Zonotope
from zonopt.polytope.zonotope import express_as_subset_sum
from zonopt.train.loss import get_facet_normal, get_vertex_on_facet, hausdorff_loss
from zonopt.metrics import hausdorff_distance
import torch
import numpy as np
import unittest


class TestHausdorffLoss(unittest.TestCase):
    def setUp(self):
        # (Z1,P1) Will achieve Hausdorff distance at type I
        generators1 = torch.tensor(
            [[0.01,0.45],[0.34,0.06],[0.12,0.51]],
            dtype=torch.float64,
            requires_grad=True,
        )
        translation1 = torch.zeros(2, requires_grad=True, dtype=torch.float64)
        self.Z1 = Zonotope(generators=generators1, translation=translation1)

        vertices1 = np.array([[0.59,0.79],[0.1,0.24],[0.29,0.22],[0.91,0.52],[0.89,0.68]])
        self.P1 = Polytope(points=vertices1)

        # (Z2, P2) Will achieve hausdorff distance at type II
        generators2 = torch.tensor(
            [[0.25, 0.33], [0.5, 0.18], [0.21, 0.18]],
            dtype=torch.float64,
            requires_grad=True,
        )
        translation2 = torch.zeros(2, requires_grad=True, dtype=torch.float64)
        self.Z2 = Zonotope(generators=generators2, translation=translation2)

        vertices2 = np.array([[1.76, 1.79], [0.1, 1.85], [0.13, 0.55]])
        self.P2 = Polytope(points=vertices2)

        # hausdorff distance
        self.haus_dist1, [self.p1], [self.q1] = hausdorff_distance(self.P1, self.Z1)
        self.haus_dist2, [self.p2], [self.q2] = hausdorff_distance(self.P2, self.Z2)

    def test_typeI_grad(self):
        loss = hausdorff_loss(self.Z1, self.P1, self.q1, self.p1)
        loss.backward()

        # Hausdorff loss should be the same as hausdorff distance
        np.testing.assert_almost_equal(loss.item(), self.haus_dist1)

        # p1 should lie on a codimension 1 face.
        halfspaces = self.P1.supporting_halfspaces(self.p1)
        self.assertTrue(len(halfspaces) == 1)

        # q1 should be a vertex of Z
        self.assertTrue(self.Z1.has_vertex(self.q1))

        eta = halfspaces[0]._a
        _, epsilon = express_as_subset_sum(self.q1, self.Z1, return_indices=False)

        # Check against Proposition 6.12
        for i in range(self.Z1.rank):
            for j in range(self.Z1.dimension):
                # Algebraic value of d \delta_q / d g_{i,j}
                target_gradient = eta[j] * epsilon[i]
                # Calculated value via loss
                loss_gradient = self.Z1.generators.grad[i][j].item()
                np.testing.assert_almost_equal(target_gradient, loss_gradient)

        for j in range(self.Z1.dimension):
            # Algebraic value of d \delta_q / d\mu_j
            target_gradient = eta[j]
            # Calculated value via loss
            loss_gradient = self.Z1.translation.grad[j].item()
            np.testing.assert_almost_equal(target_gradient, loss_gradient)

    def test_typeII_grad(self):
        loss = hausdorff_loss(self.Z2, self.P2, self.q2, self.p2)
        loss.backward()

        # Hausdorff loss should be the same as hausdorff distance
        np.testing.assert_almost_equal(loss.item(), self.haus_dist2)

        # q2 should lie on a codimension 1 face.
        halfspaces = self.Z2.supporting_halfspaces(self.q2)
        self.assertTrue(len(halfspaces) == 1)

        # p2 should be a vertex of P2
        self.assertTrue(self.P2.has_vertex(self.p2))

        vertex_on_facet = get_vertex_on_facet(self.Z2, halfspaces[0])
        _, epsilon = express_as_subset_sum(
            vertex_on_facet, self.Z2, return_indices=False
        )

        # Check against Proposition 6.16
        for i in range(self.Z2.rank):
            for j in range(self.Z2.dimension):
                Z2 = self.Z2.copy(requires_grad=True)
                eta = get_facet_normal(Z2, halfspaces[0])
                target_gradient = -eta[j] * epsilon[i]
                for jp in range(self.Z2.dimension):
                    Z2.generators.grad = None
                    etajp = eta[jp]
                    etajp.backward(retain_graph=True)
                    d_etajp = Z2.generators.grad[i][j].item()
                    target_gradient += d_etajp * (self.p2[jp] - vertex_on_facet[jp])

                loss_gradient = self.Z2.generators.grad[i][j].item()
                np.testing.assert_almost_equal(loss_gradient, target_gradient.item())

        for j in range(self.Z2.dimension):
            eta = get_facet_normal(self.Z2, halfspaces[0])
            target_gradient = -eta[j]

            loss_gradient = self.Z2.translation.grad[j].item()
            np.testing.assert_almost_equal(loss_gradient, target_gradient.item())

    def test_loss_matches_higher_dimension(self):
        P = Polytope.random(10, 4, seed=2)
        Z = Zonotope.random(6, 4, use_torch=True, requires_grad=False, seed=2)

        haus_dist, pp, qq = hausdorff_distance(P, Z)
        loss = hausdorff_loss(Z, P, qq[0], pp[0])

        np.testing.assert_almost_equal(haus_dist, loss.item())

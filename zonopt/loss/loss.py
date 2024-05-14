from zonopt.metrics import hausdorff_distance
from zonopt.polytope import Polytope, Zonotope
from zonopt.polytope.zonotope import express_as_subset_sum
import numpy as np
import torch


def hausdorff_loss(Z: Zonotope, P: Polytope, q: np.ndarray, p: np.ndarray):
    """
    Calculate the hausdorff distance in a differentiable way (w.r.t. Zonotope
    parameters). Must specify points p in P and q in Z that achieve the distance.

    Parameters:
    -----------
    Z: Zonotope
        Generators must be torch tensors with grad.
    P: Polytope
    q: np.ndarray
        A point in Z and
    p: np.ndarray
        a point in P such that d_H(P,Z) = d(p,q)
    """

    if Z.has_vertex(q):
        return _hausdorff_loss_typeI(Z, P, q, p)
    elif P.has_vertex(p):
        return _hausdorff_loss_typeII(Z, P, q, p)
    else:
        raise ValueError(
            "At least one of p or q must be a vertex to differentiably calculate the hausdorff distance."
        )


def _hausdorff_loss_typeI(Z: Zonotope, P: Polytope, q: np.ndarray, p: np.ndarray):
    """
    Implementation of above when q is a vertex of Z.
    """
    is_subset_sum, subset = express_as_subset_sum(
        q,
        list(Z.generators.detach().numpy()),
        base=Z.translation.detach().numpy(),
        return_indices=True,
    )
    if not is_subset_sum:
        raise ValueError("Specified point is not a vertex of Z")

    # q = Q*epsilon + mu
    control_pt = 1.0 * Z.translation
    for i in subset:
        control_pt += Z.generators[i]

    if P.has_vertex(p):
        return torch.norm(control_pt - p)

    halfspaces = P.supporting_halfspaces(p)
    diff = torch.zeros(len(q))
    for H in halfspaces:
        eta = torch.tensor(H._a)
        c = torch.tensor(H._c)
        diff += (eta @ control_pt - c) * eta

    return torch.norm(diff)


def _hausdorff_loss_typeII(Z: Zonotope, P: Polytope, q: np.ndarray, p: np.ndarray):
    """
    Implementation of above when p is a vertex of P.
    """
    pass

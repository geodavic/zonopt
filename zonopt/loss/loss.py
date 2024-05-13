from zonopt.metrics import hausdorff_distance
from zonopt.polytope import Polytope, Zonotope
import numpy as np
import torch


def hausdorff_loss(Z: Zonotope, P: Polytope, q: np.ndarray, p: np.ndarray):
    """
    Calculate the hausdorff distance in a differentiable way. Must specify
    points p in P and q in Z that achieve the distance.

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
    pass


def _hausdorff_loss_typeII(Z: Zonotope, P: Polytope, q: np.ndarray, p: np.ndarray):
    """
    Implementation of above when p is a vertex of P.
    """
    pass

from zonopt.metrics import hausdorff_distance
from zonopt.polytope import Polytope, Zonotope, Halfspace
from zonopt.polytope.zonotope import express_as_subset_sum
from zonopt.polytope.comparisons import almost_equal
from zonopt.polytope.exceptions import GeometryError
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
    Implementation of hausdorff loss when q is a vertex of Z.
    """
    is_subset_sum, subset = express_as_subset_sum(
        q,
        list(Z.generators.detach().numpy()),
        base=Z.translation.detach().numpy(),
        return_indices=True,
    )
    if not is_subset_sum:
        raise GeometryError("Specified point is not a vertex of Z")

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
    Implementation of hausdorff loss when p is a vertex of P and q is not a vertex of Z.
    """

    halfspaces = Z.supporting_halfspaces(q)
    diff = torch.zeros(len(q))
    for H in halfspaces:
        facet_subset = get_facet_generators(Z, H)
        sample_vertex = get_vertex_on_facet(Z, H)

        if sample_vertex is None:
            raise GeometryError(
                "Could not sample a vertex from Z and the given halfspace"
            )

        is_subset_sum, subset = express_as_subset_sum(
            sample_vertex,
            list(Z.generators.detach().numpy()),
            base=Z.translation.detach().numpy(),
            return_indices=True,
        )
        if not is_subset_sum:
            raise GeometryError("Vertex found on Z can't be expressed as a subset sum.")

        # Express sample vertex using Z generators and translation
        sample_vertex_torch = 1.0 * Z.translation
        for i in subset:
            sample_vertex_torch += Z.generators[i]

        # Calculate eta, the normal to H, in terms of Z generators
        # Sign of eta won't matter.
        slicer = lambda i, g: torch.cat([g[:i], g[(i + 1) :]])
        submatrix_generator = lambda i: torch.stack(
            [slicer(i, Z.generators[j]) for j in subset]
        )
        eta = torch.stack(
            [
                (-1) ** (i + 1) * torch.det(submatrix_generator(i))
                for i in range(Z.dimension)
            ]
        )
        eta = eta / torch.norm(eta)

        # Add residual
        diff += (eta @ (p - sample_vertex_torch)) * eta

    return torch.norm(diff)


def get_vertex_on_facet(Z: Zonotope, H: Halfspace):
    """
    Get a vertex of Z lying on the boundary of a supporting
    halfspace of Z.
    """
    for v in Z.vertices:
        if H.boundary.contains(v):
            return v


def get_facet_generators(Z: Zonotope, H: Halfspace):
    """
    Get the indices of generators of Z that are orthogonal to
    a supporting halfspace of Z.
    """
    gens = []
    for ids, g in enumerate(Z.generators):
        if almost_equal(H.a @ g, 0):
            gens += [idx]
    return gens

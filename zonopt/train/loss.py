from zonopt.polytope import Polytope, Zonotope, Halfspace
from zonopt.polytope.comparisons import almost_equal
import numpy as np
import torch


def hausdorff_loss(Z: Zonotope, P: Polytope, q: np.ndarray, p: np.ndarray):
    """
    Calculate the hausdorff distance in a differentiable way (w.r.t. Zonotope
    parameters). Must specify points p in P and q in Z that achieve the distance,
    at least one of which is a vertex of its polytope.

    Let epsilon = Lift(q) be the (unique, since Z is stable) point of the unit
    cube with Z(epsilon) = Q^T epsilon + mu = q. The loss is

        || Q^T epsilon + mu - p ||_2

    with epsilon and p held constant. In both cases this has the same value
    and the same gradient as the local expressions for d_P in the paper:

    - Type I (q a vertex of Z, p = Pi(q, P)): d(q, P) = min_{p' in P} ||q - p'||
      and p' ranges over a set independent of Z, so by Danskin's theorem the
      gradient of d(q(Z), P) is that of ||q(Z) - p|| with p fixed
      (Proposition `grad1`).
    - Type II (p a vertex of P, q = Pi(p, Z)): d(p, Z) = min_{e in [0,1]^n}
      ||p - Q^T e - mu||. The feasible set [0,1]^n is independent of (Q, mu),
      so again by Danskin the gradient is that of ||p - Q^T epsilon - mu|| with
      epsilon fixed. This agrees with Proposition `grad2` (the terms involving
      d eta / d g_ij collapse to -eta_j epsilon_i for the fractional
      coordinates of epsilon) and extends it to faces of any dimension.

    This avoids describing the face containing q by supporting hyperplanes,
    which is where the previous implementation failed for dimension > 2: the
    supporting hyperplanes of a face overdetermine it, their normals are not
    orthonormal (so sum_i <eta_i, u - x> eta_i is not u - x for codimension >= 2),
    and the number of hyperplanes needed is the codimension, not the dimension,
    of the face.

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
    if not Z.is_torch:
        raise ValueError("Z must be a torch zonotope to compute the hausdorff loss")

    if not (Z.has_vertex(q) or P.has_vertex(p)):
        raise ValueError(
            "At least one of p or q must be a vertex to differentiably calculate the hausdorff distance."
        )

    epsilon = Z.lift(q)
    control_pt = Z.generators.T @ torch.as_tensor(epsilon, dtype=Z.generators.dtype)
    control_pt = control_pt + Z.translation
    return torch.norm(control_pt - torch.as_tensor(p, dtype=control_pt.dtype))


def get_facet_normal(Z: Zonotope, H: Halfspace):
    """
    Get a unit normal to a supporting halfspace on a Zonotope.
    Same numerical value as H.a (up to sign), but differentiable in terms of
    the generators of Z. See Proposition `grad2` of the paper. Only valid
    for a general position zonotope, where exactly d-1 generators are
    orthogonal to the facet normal.
    """
    facet_subset = get_facet_generators(Z, H)
    if len(facet_subset) != Z.dimension - 1:
        raise ValueError(
            f"Expected {Z.dimension - 1} generators parallel to the facet, found {len(facet_subset)}. Is Z in general position?"
        )
    slicer = lambda i, g: torch.cat([g[:i], g[(i + 1) :]])
    submatrix_generator = lambda i: torch.stack(
        [slicer(i, Z.generators[j]) for j in facet_subset]
    )
    eta = torch.stack(
        [
            (-1) ** (i + 1) * torch.det(submatrix_generator(i))
            for i in range(Z.dimension)
        ]
    )
    eta = eta / torch.norm(eta)
    return eta


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
    for idx, g in enumerate(Z.generators):
        if Z.is_torch:
            g = g.detach().numpy()
        if almost_equal(H.a @ g, 0):
            gens += [idx]
    return gens

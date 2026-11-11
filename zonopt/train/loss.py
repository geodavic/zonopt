"""
Differentiable Hausdorff loss between a polytope P and a zonotope Z.

Two implementations of the same loss live in this module. Given a pair of
points (p, q) in P x Z achieving the Hausdorff distance, at least one of which
is a vertex, both return a torch scalar equal to ||p - q|| whose gradient with
respect to the generators and translation of Z is a (Clarke) subgradient of
d_P(Z) = d(P, Z).

1. `hausdorff_loss_hyperplane` is the formulation in the paper
   (arXiv:2407.13125, Section "Computing the Clarke subdifferential"). The
   distance is written as the distance from a point to an affine subspace
   W = Aff(F), with W given as an intersection of hyperplanes. For a face of
   Z the hyperplane normals are expressed through the (d-1)x(d-1) minors of the
   facet generators (Proposition `grad2`) so that they are differentiable in
   the parameters of Z. One correction to the paper is applied: the residual
   is N^T (N N^T)^{-1} (N u - c) rather than N^T (N u - c), since the normals
   are not orthonormal when the face has codimension >= 2. Without the Gram
   matrix N N^T the formula is only correct in codimension 1, which is why the
   original implementation worked in dimension 2 and nowhere else.

2. `hausdorff_loss` is a simplification found after the paper was written.
   Since Z is the image of the fixed cube [0,1]^n under eps -> Q^T eps + mu,
   the minimization defining d(p, Z) can be pulled back to the cube, where the
   feasible set no longer depends on (Q, mu). Danskin's theorem then gives the
   gradient as that of ||Q^T eps + mu - p|| with eps = Lift(q) held fixed; the
   same expression covers the case where q is a vertex of Z (eps in {0,1}^n).
   Uniqueness of the lift, i.e. stability of Z, is exactly the hypothesis that
   makes this valid. No faces, normals or determinants are needed, and the
   resulting subgradient is the vector (q - p)/||q - p|| (x) (Lift(q), 1) that
   already appears in the paper's feasibility cone.

The two agree in value and gradient to machine precision (see
`tests/test_train/test_loss.py`). The trainer uses `hausdorff_loss`; the
hyperplane version is kept as the reference implementation of the paper.
"""
from zonopt.polytope import Polytope, Zonotope, Halfspace
from zonopt.polytope.comparisons import almost_equal
from zonopt.polytope.exceptions import GeometryError
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

    See the module docstring for how this relates to the hyperplane
    formulation in the paper (`hausdorff_loss_hyperplane`).

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


# ---------------------------------------------------------------------------
# Hyperplane (determinant) formulation of the loss, following the paper.
# ---------------------------------------------------------------------------


def hausdorff_loss_hyperplane(Z: Zonotope, P: Polytope, q: np.ndarray, p: np.ndarray):
    """
    Same value and gradient as `hausdorff_loss`, but computed the way the paper
    does it: as the distance between a point u and an affine subspace
    W = Aff(F), where W is written as an intersection of hyperplanes
    H_i = { y : <eta_i, y> = c_i } with unit normals eta_i.

    Writing N for the matrix with rows eta_i and c for the vector of offsets,
    the residual u - Pi(u, W) lies in span(eta_i), and

        u - Pi(u, W) = N^T (N N^T)^{-1} (N u - c)

    so that  d(u, W)^2 = (N u - c)^T (N N^T)^{-1} (N u - c).

    The Gram matrix N N^T is the identity only when the eta_i are orthonormal,
    which is automatic for a single hyperplane (codimension 1, the only case
    that arises in dimension 2) but not in general. Dropping it is the error
    behind the original dimension > 2 failure.

    - Type I (q a vertex of Z): u = Q^T e + mu is differentiable and W = Aff(F_p)
      is a fixed subspace of P, so N and c are constants.
    - Type II (p a vertex of P): u = p is constant and W = Aff(F_q) is a face
      of Z. Each eta_i is the normal of a facet of Z containing F_q, computed
      from the (d-1)x(d-1) minors of the facet's generators (Proposition
      `grad2`), and c_i = <eta_i, v> for a vertex v = Q^T e + mu of F_q. Both
      are differentiable in (Q, mu).

    In both cases a minimal determining set of hyperplanes (codim(W) of them
    with linearly independent normals) is selected from the supporting
    hyperplanes, which in general overdetermine the face.
    """
    if not Z.is_torch:
        raise ValueError("Z must be a torch zonotope to compute the hausdorff loss")
    dtype = Z.generators.dtype

    if Z.has_vertex(q):
        # Type I: moving point, fixed subspace (a face of P).
        halfspaces = _independent_supporting_halfspaces(P, p)
        N = torch.as_tensor(np.array([H._a for H in halfspaces]), dtype=dtype)
        c = torch.as_tensor(np.array([H._c for H in halfspaces]), dtype=dtype)
        u = _torch_point(Z, Z.lift(q))
    elif P.has_vertex(p):
        # Type II: fixed point, moving subspace (a face of Z).
        halfspaces = _independent_supporting_halfspaces(Z, q)
        v = get_vertex_on_face(Z, halfspaces)
        v_torch = _torch_point(Z, Z.lift(v))
        etas = [get_facet_normal(Z, H) for H in halfspaces]
        N = torch.stack(etas)
        c = N @ v_torch
        u = torch.as_tensor(p, dtype=dtype)
    else:
        raise ValueError(
            "At least one of p or q must be a vertex to differentiably calculate the hausdorff distance."
        )

    residual = N @ u - c
    gram = N @ N.T
    return torch.sqrt(residual @ torch.linalg.solve(gram, residual))


def _torch_point(Z: Zonotope, epsilon: np.ndarray):
    """Q^T epsilon + mu as a torch expression in the parameters of Z."""
    eps = torch.as_tensor(epsilon, dtype=Z.generators.dtype)
    return Z.generators.T @ eps + Z.translation


def _independent_supporting_halfspaces(X: Polytope, x: np.ndarray):
    """
    Select, from the supporting halfspaces of X at x, a subset whose normals
    are linearly independent and span the normal space of the face containing
    x. There are codim(face) of them.
    """
    supporting = X.supporting_halfspaces(x)
    chosen = []
    normals = []
    for H in supporting:
        candidate = normals + [H._a]
        if np.linalg.matrix_rank(np.array(candidate), tol=1e-8) == len(candidate):
            chosen.append(H)
            normals.append(H._a)
    codim = X.dimension - X.face_dimension(x)
    if len(chosen) != codim:
        raise GeometryError(
            f"Found {len(chosen)} independent supporting hyperplanes at {x}, expected {codim}."
        )
    return chosen


def get_vertex_on_face(Z: Zonotope, halfspaces):
    """
    Get a vertex of Z lying on the boundary of every given supporting
    halfspace of Z, i.e. a vertex of the face they cut out.
    """
    for v in Z.vertices:
        if all(H.boundary.contains(v) for H in halfspaces):
            return v
    raise GeometryError("No vertex of Z lies on all of the given hyperplanes.")

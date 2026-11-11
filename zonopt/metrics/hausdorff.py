from scipy.optimize import linprog, lsq_linear
from qpsolvers import solve_qp
import numpy as np
import scipy as sp
import scipy.sparse
import osqp
import warnings
from zonopt.config import global_config
from zonopt.polytope import Polytope, Zonotope, UnitBall, Hyperplane
from zonopt.polytope.utils import express_point_as_convex_sum
from zonopt.metrics.exceptions import HausdorffError


def _distance_to_zonotope_l2(x: np.ndarray, Z: Zonotope):
    """
    Find the distance from x to a zonotope Z = { Q^T e + mu : e in [0,1]^n }
    in L^2 norm, by solving the bounded least squares problem

        min_{e in [0,1]^n} || Q^T e + mu - x ||_2

    in the cube coordinates e. This has n variables regardless of how many
    vertices Z has, and the minimizer e is the lift of the projection.

    Returns
    -------
    dist: float
    proj: np.ndarray
        The projection of x onto Z.
    epsilon: np.ndarray
        Cube coordinates of the projection, i.e. its lift. Entries within
        comparison_epsilon of 0 or 1 are snapped to those values.
    """
    generators, translation = Z._numpy_parameters()
    x = np.asarray(x, dtype=np.float64)
    result = lsq_linear(
        generators.T, x - translation, bounds=(0, 1), method="bvls", tol=1e-14
    )
    epsilon = result.x
    snap = global_config.comparison_epsilon
    epsilon[np.abs(epsilon) < snap] = 0.0
    epsilon[np.abs(epsilon - 1) < snap] = 1.0
    proj = generators.T @ epsilon + translation
    dist = np.linalg.norm(proj - x)
    if dist < snap:
        dist, proj = 0.0, x
    return dist, proj, epsilon


class _VertexProjectionQP:
    """
    Projection onto a fixed polytope P = conv(V) as a quadratic program in
    (x, lambda):  min 1/2 ||x||^2 - <p, x>  s.t.  x = V^T lambda,
    sum(lambda) = 1, lambda >= 0  (see https://arxiv.org/abs/1401.1434).

    Only the linear term depends on the point p, so the problem is set up
    once per polytope and re-solved (warm started) for every new point. This
    avoids rebuilding the sparse matrices for each call.
    """

    def __init__(self, P: Polytope):
        V = P.vertices
        self.d, self.m = V.shape[1], len(V)
        d, m = self.d, self.m
        Pmat = scipy.sparse.block_diag(
            [scipy.sparse.eye(d), scipy.sparse.csc_matrix((m, m))], format="csc"
        )
        A = scipy.sparse.vstack(
            [
                scipy.sparse.hstack([scipy.sparse.eye(d), -scipy.sparse.csc_matrix(V.T)]),
                scipy.sparse.hstack([scipy.sparse.csc_matrix((1, d)), np.ones((1, m))]),
                scipy.sparse.hstack([scipy.sparse.csc_matrix((m, d)), scipy.sparse.eye(m)]),
            ],
            format="csc",
        )
        lower = np.concatenate([np.zeros(d), [1.0], np.zeros(m)])
        upper = np.concatenate([np.zeros(d), [1.0], np.full(m, np.inf)])
        cfg = global_config.qp_config
        self.problem = osqp.OSQP()
        self.problem.setup(
            Pmat,
            np.zeros(d + m),
            A,
            lower,
            upper,
            verbose=False,
            polish=bool(cfg.polish),
            eps_abs=cfg.eps_abs,
            eps_rel=cfg.eps_rel,
            max_iter=cfg.max_iter,
        )

    def solve(self, x: np.ndarray):
        self.problem.update(q=np.concatenate([-x, np.zeros(self.m)]))
        result = self.problem.solve()
        if result.x is None or np.any(np.isnan(result.x)):
            raise HausdorffError("Quadratic program for l2 distance did not succeed.")
        proj = result.x[: self.d]
        coeffs = result.x[self.d :]
        return np.linalg.norm(proj - x), proj, coeffs


def _vertex_projection_qp(P: Polytope):
    qp = getattr(P, "_vertex_projection_qp", None)
    if qp is None or qp.m != len(P.vertices):
        qp = _VertexProjectionQP(P)
        P._vertex_projection_qp = qp
    return qp


def _distance_to_polytope_l2(x: np.array, P: Polytope):
    """
    Find the distance from x to P in L^2 norm.
    Uses a quadratic program to find the projection of
    x onto P. See https://arxiv.org/abs/1401.1434.
    For a Zonotope, the problem is solved in cube coordinates instead
    (see `_distance_to_zonotope_l2`).

    Parameters:
    -----------
    x: np.ndarray
        The input point.
    P: Polytope
        The input polytope.

    Returns:
    --------
    dist: float
        The distance d(x,P) under L^2. This is the
        distance between x and the projection of x
        onto P.
    proj: np.ndarray
        The projection of x onto P.
    coeffs: np.ndarray
        The coefficients of the sum representing p as a
        convex sum of vertices of P (for a Zonotope: the
        cube coordinates of the projection).
    """

    if not isinstance(x, np.ndarray) or x.dtype != np.float64:
        x = np.array(x, np.float64)

    if isinstance(P, Zonotope):
        return _distance_to_zonotope_l2(x, P)

    if P.contains(x):
        # `contains` is tolerance based while the LP below is exact, so a point
        # just outside P can pass the first check and fail the second. In that
        # case fall through to the quadratic program rather than failing.
        coeffs = express_point_as_convex_sum(x, P.vertices)
        if coeffs is not None:
            return 0, x, coeffs

    return _vertex_projection_qp(P).solve(x)


def _distance_to_polytope_l1_infty(x: np.ndarray, P: Polytope, metric: int = 2):
    """
    Find the distance from x to P in L^1 or L^\inft norm.
    Uses a linear program. See https://arxiv.org/abs/1401.1434.

    Parameters:
    -----------
    x: np.ndarray
        The input point.
    P: Polytope
        The input polytope.

    Returns:
    --------
    dist: float
        The distance d(x,P) under L^2. This is the
        distance between x and the projection of x
        onto P.
    proj: np.ndarray
        The projection of x onto P.
    coeffs: np.ndarray
        The coefficients of the sum representing p as a
        convex sum of vertices of P.
    """

    B_vertices = UnitBall(len(x), p=metric)
    lambda_coefs = np.concatenate(
        [P_vertices.T, [np.ones(len(P_vertices))], [np.zeros(len(P_vertices))]],
        axis=0,
    )
    mu_coefs = np.concatenate(
        [B_vertices.T, [np.zeros(len(B_vertices))], [np.ones(len(B_vertices))]],
        axis=0,
    )

    rho_coefs = np.zeros(len(x) + 2)
    rho_coefs[-1] = -1
    rho_coefs = np.expand_dims(rho_coefs, axis=0).T

    Aeq = np.concatenate([lambda_coefs, mu_coefs, rho_coefs], axis=1)
    beq = np.concatenate([x, [1, 0]])
    c = np.zeros(len(Aeq[0]))
    c[-1] = 1

    prog = linprog(c, A_ub=None, b_ub=None, A_eq=Aeq, b_eq=beq)
    if prog.status != 0:
        raise Exception("LP to determine Hausdorff distance is infeasible or unbounded")

    dist = prog.x[-1]
    coeffs = prog.x[: len(P_vertices)]
    proj = np.sum([c * P_vertices[i] for c in coeffs], axis=0)
    return dist, sol, coeffs


def distance_to_polytope(x: np.ndarray, P: Polytope, metric: int = 2):
    """
    Compute the distance between a point x and a polytope P
    under the specified metric. Only metrics 1,2 and \infty are supported
    currently.

    Parameters:
    -----------
    x: np.ndarray
        The input point.
    P: Polytope
        The input polytope.

    Returns:
    --------
    dist: float
        The distance d(x,P) under L^2. This is the
        distance between x and the projection of x
        onto P.
    proj: np.ndarray
        The projection of x onto P.
    coeffs: np.ndarray
        The coefficients of the sum representing p as a
        convex sum of vertices of P.

    """

    if metric == 1 or metric == np.infty:
        return _distance_to_polytope_l1_infty(x, P, metric=metric)
    elif metric == 2:
        return _distance_to_polytope_l2(x, P)
    else:
        raise NotImplementedError("Only L^p metrics for p = 1,2,np.infty supported.")


def distance_to_hyperplane(x: np.ndarray, H: Hyperplane, metric: int = 2):
    """
    Compute the distance between a point x and a hyperplane H.
    Only metric 2 is supported currently.

    Parameters:
    -----------
    x: np.ndarray
        The input point.
    P: Polytope
        The input polytope.

    Returns:
    --------
    dist: float
        The distance d(x,H) under L^2. This is the
        distance between x and the projection of x
        onto H.
    """
    if metric == 2:
        dist = x @ H._a - H._c
    else:
        raise NotImplementedError("Only L^2 metrics supported.")
    return dist


def hausdorff_distance(
    P: Polytope, Q: Polytope, threshold: float = 1.0, metric: int = 2
):
    """
    Compute the Hausdorff distance between P and Q:

    d(P,Q) = max( min_{x\in Q}d(x,P), \min_{y\in P}d(y,Q) )

    Returns the distance, a pair of points p \in P, q \in Q where the distance
    is achieved and at which pairs of points that distance is achieved (within the
    specified threshold)

    Parameters:
    ----------
    P: Polytope
        Input polytope
    Q: Polytope
        Input polyotpe
    threshold: float
        Threshold to determine the multiplicity (see above).

    Returns:
    -------
    dist: float
        The Hausdorff distance d(P,Q)
    p: list[np.ndarray]
        A list of points in P and
    q: list[np.ndarray]
        A list of points in Q such that each pair (p_i,q_i) achieves the Hausdorff distance
        within the specified threshold.
    """

    def one_sided(source: Polytope, target: Polytope):
        """
        Distances from the vertices of `source` to `target`, solving a
        projection problem only where cheap bounds cannot settle it.

        For a vertex v of `source`, with facets a_i . x <= c_i of `target`:
          lower bound  L(v) = max_i (a_i . v - c_i)_+
          upper bound  U(v) = distance from v to the nearest vertex of `target`
        If the projection of v onto the hyperplane of its most violated facet
        lies in `target`, that projection is the nearest point and the
        distance equals L(v) exactly (the common case). Otherwise v only needs
        a solve if U(v) >= threshold * best, where best is the largest
        distance established so far; vertices are visited in order of
        decreasing L so that `best` grows quickly.
        """
        V = source.vertices
        eq = target.equations
        normals, offsets = eq[:, :-1], eq[:, -1]
        violations = V @ normals.T + offsets  # (|V|, facets)
        worst = violations.argmax(axis=1)
        lower = np.maximum(violations[np.arange(len(V)), worst], 0)

        # Exact shortcut: projection onto the most violated facet's hyperplane.
        foot = V - lower[:, None] * normals[worst]
        inside = np.all(
            foot @ normals.T + offsets < global_config.comparison_epsilon, axis=1
        )

        diffs = V[:, None, :] - target.vertices[None, :, :]
        upper = np.sqrt((diffs**2).sum(axis=-1)).min(axis=1)

        results = []
        best = lower.max()
        for i in np.argsort(-lower):
            if inside[i]:
                dist, proj = lower[i], foot[i]
            elif upper[i] >= threshold * best:
                dist, proj, _ = distance_to_polytope(V[i], target, metric=metric)
            else:
                continue
            best = max(best, dist)
            results.append((dist, V[i], proj))
        return sorted(results, key=lambda x: -x[0])

    distances_P = one_sided(P, Q)
    distances_Q = [(d, proj, q) for d, q, proj in one_sided(Q, P)]

    haus_dist = max(distances_P[0][0], distances_Q[0][0])

    points_P = []
    points_Q = []
    for d, p, q in distances_P + distances_Q:
        if d >= threshold * haus_dist:
            points_P += [p]
            points_Q += [q]

    return haus_dist, points_P, points_Q

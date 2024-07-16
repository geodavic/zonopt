from scipy.optimize import linprog
from qpsolvers import solve_qp
import numpy as np
import scipy as sp
import warnings
from zonopt.config import global_config
from zonopt.polytope import Polytope, UnitBall, Hyperplane
from zonopt.polytope.utils import express_point_as_convex_sum
from zonopt.metrics.exceptions import HausdorffError


def _distance_to_polytope_l2(x: np.array, P: Polytope):
    """
    Find the distance from x to P in L^2 norm.
    Uses a quadratic program to find the projection of
    x onto P. See https://arxiv.org/abs/1401.1434.

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

    if not isinstance(x, np.ndarray) or x.dtype != np.float64:
        x = np.array(x, np.float64)

    if P.contains(x):
        coeffs = express_point_as_convex_sum(x, P.vertices)
        if coeffs is not None:
            return 0, x, coeffs
        else:
            raise HausdorffError("Linear program for l2 distance did not succeed.")

    P_vertices = P.vertices

    Q = sp.linalg.block_diag(
        np.eye(len(x)), np.zeros((len(P_vertices), len(P_vertices)))
    )
    q = -np.concatenate([x, np.zeros(len(P_vertices))])
    A = np.concatenate([np.eye(len(x)), -P_vertices.T], axis=1)
    A = np.concatenate([A, [[0] * len(x) + [1] * len(P_vertices)]], axis=0)
    b = np.array([0] * len(x) + [1], np.float64)
    lb = np.array([-np.inf] * len(x) + [0] * len(P_vertices), np.float64)
    ub = np.array([np.inf] * (len(x) + len(P_vertices)), np.float64)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=r"qpsolvers*")
        sol = solve_qp(
            Q,
            q,
            A=A,
            b=b,
            lb=lb,
            ub=ub,
            solver=global_config.qp_config.solver,
            polish=global_config.qp_config.polish,
            eps_abs=global_config.qp_config.eps_abs,
            eps_rel=global_config.qp_config.eps_rel,
            max_iter=global_config.qp_config.max_iter,
        )

    if sol is None:
        raise HausdorffError("Quadratic program for l2 distance did not succeed.")

    proj = sol[: len(x)]
    coeffs = sol[len(x) :]
    dist = np.linalg.norm(proj - x)
    return dist, proj, coeffs


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

    distances_P = []
    for p in P.vertices:
        dist, projp, _ = distance_to_polytope(p, Q, metric=metric)
        distances_P += [[dist, p, projp]]
    distances_P = sorted(distances_P, key=(lambda x: -x[0]))

    distances_Q = []
    for q in Q.vertices:
        dist, projq, _ = distance_to_polytope(q, P, metric=metric)
        distances_Q += [[dist, projq, q]]
    distances_Q = sorted(distances_Q, key=(lambda x: -x[0]))

    haus_dist = max(distances_P[0][0], distances_Q[0][0])

    points_P = []
    points_Q = []
    for d, p, q in distances_P + distances_Q:
        if d >= threshold * haus_dist:
            points_P += [p]
            points_Q += [q]

    return haus_dist, points_P, points_Q

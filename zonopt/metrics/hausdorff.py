from scipy.optimize import linprog
from qpsolvers import solve_qp
import numpy as np
import scipy as sp
from zonopt.config import global_config
from zonopt.polytope import Polytope, UnitBall, Hyperplane


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



import numpy as np
from typing import Any
from zonopt.polytope.comparisons import almost_equal
from zonopt.polytope.exceptions import GeometryError
from zonotope.todo import GeorgePleaseImplement
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from itertools import combinations
import torch


def subsets(arr):
    subsets = []
    [subsets.extend(list(combinations(arr, n))) for n in range(len(arr) + 1)]
    return subsets


def binary_lists(n):
    """
    Return all binary lists of length n.
    """
    return [[i in s for i in range(n)] for s in subsets(list(range(n)))]


def affine_from_vertices(vertices: np.ndarray):
    """
    Get the affine map associated to a zonotope given its vertices.
    """
    if not is_centrally_symmetric(vertices):
        raise GeometryError(
            "Given vertices are not centrally symmetric, can't compute zonotope generators."
        )

    raise GeorgePleaseImplement(
        "creating the generators from the vertices of a zonotope"
    )


def hull_from_affine(generators: Any, translation: Any):
    """
    Get the convex hull of a zonotope given by an affine map.

    TODO: make this more memory efficient, not generating all cubical vertices.
    """
    if isinstance(translation, torch.Tensor):
        translation_ = translation.detach().numpy()
    else:
        translation_ = np.array(translation)

    if isinstance(generators, torch.Tensor):
        generators_ = generators.detach().numpy()
    else:
        generators_ = np.array(generators)

    cubical_vertices = [np.zeros(len(generators_[0])) + translation_]
    for subset in subsets(generators_)[1:]:
        cubical_vertices.append((np.sum(subset, axis=0) + translation_))
    cubical_vertices = np.array(cubical_vertices)
    return ConvexHull(cubical_vertices)


def translate_points(points: np.ndarray, translation: Any):
    if isinstance(translation, torch.Tensor):
        translation_ = translation.detach().numpy()
    else:
        translation_ = np.array(translation)
    return points + translation_


def is_centrally_symmetric(points: np.ndarray):
    """
    Check if a collection of points is centrally symmetric.
    """

    barycenter = np.sum(points, axis=0) / len(points)
    for pt1 in points:
        reflected = 2 * barycenter - pt1
        found_reflection = False
        for pt2 in points:
            if almost_equal(pt1, pt2):
                found_reflection = True
                break
        if not found_reflection:
            return False
    return True


def express_point_as_convex_sum(x: np.ndarray, points: np.ndarray):
    """
    Express x as a convex sum of the elements of `points`.

    Returns vector of coefficients \mu_i such that
    \sum_i \mu_i p_i = x
    where \sum_i \mu_i = 1 and \mu_i >= 0

    If no solution exists, return None.
    """

    Aeq = np.vstack((points.T, [1] * len(points)))
    beq = np.hstack((x, 1))
    c = np.zeros(len(points))

    lp = linprog(c, A_eq=Aeq, b_eq=beq)

    if lp.status != 0:
        return None

    return lp.x

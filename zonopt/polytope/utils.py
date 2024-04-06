import numpy as np
from zonopt.polytope.comparisons import almost_equal
from zonopt.polytope.exceptions import GeometryError
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from itertools import combinations


def subsets(arr):
    subsets = []
    [subsets.extend(list(combinations(arr, n))) for n in range(len(arr) + 1)]
    return subsets


def affine_from_vertices(vertices: np.ndarray):
    """
    Get the affine map associated to a zonotope given its vertices.
    """
    if not is_centrally_symmetric(vertices):
        raise GeometryError(
            "Given vertices are not centrally symmetric, can't compute zonotope generators."
        )

    raise NotImplementedError(
        "Creating the generators from the vertices of a zonotope is not yet supported"
    )


def vertices_from_affine(generators: np.ndarray, translation: np.ndarray):
    """
    Get the vertices of a zonotope given by an affine map.

    TODO: make this more memory efficient, not generating all cubical vertices.
    """
    cubical_vertices = [np.zeros(len(generators[0]))]
    for subset in subsets(generators):
        cubical_vertices.append((np.sum(subset, axis=0) + translation))
    cubical_vertices = np.array(cubical_vertices)

    return cubical_vertices[ConvexHull(cubical_vertices).vertices]


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

    Aeq = np.vstack((points.T, [1]*len(points)))
    beq = np.hstack((x,1))
    c = np.zeros(len(points))

    lp = linprog(c,A_eq=Aeq,b_eq=beq)
    
    if lp.status != 0:
        return None

    return lp.x

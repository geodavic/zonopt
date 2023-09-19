import numpy as np
from zonopt.polytope.tolerance import almost_equal, _log_global_epsilon
from zonopt.polytope.errors import GeometryError
from scipy.spatial import ConvexHull
from itertools import combinations


def subsets(arr) -> list:
    subsets = []
    [subsets.extend(list(combinations(arr, n))) for n in range(len(arr))]
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

    barycenter = np.sum(vertices, axis=0) / len(vertices)
    for pt1 in vertices:
        reflected = 2 * barycenter - pt1
        found_reflection = False
        for pt2 in vertices:
            if almost_equal(pt1, pt2):
                found_reflection = True
                break
    return found_reflection

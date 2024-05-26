from typing import List
from zonopt.polytope import Polytope
import numpy as np


def get_bounds(polytopes: List[Polytope]):
    """
    Get the bounds on a Polytope or list of polytopes.
    """
    if len(polytopes) > 1:
        minmax = np.array([list(get_bounds([P])) for P in polytopes])
        max_xy = list(np.max(minmax[:, 0, :], axis=0))
        min_xy = list(np.min(minmax[:, 1, :], axis=0))
        return max_xy, min_xy

    P = polytopes[0]
    if P.dimension != 2:
        raise ValueError("Cannot plot polytopes in dimension other than two.")

    max_xy = [-np.infty, -np.infty]
    min_xy = [np.infty, np.infty]
    for pt in P.vertices:
        for i in range(P.dimension):
            max_xy[i] = max(max_xy[i], pt[i])
            min_xy[i] = min(min_xy[i], pt[i])
    return max_xy, min_xy

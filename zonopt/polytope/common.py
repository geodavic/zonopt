import numpy as np
from zonopt.polytope import Polytope, Zonotope
from zonopt.utils import all_subsets


def UnitBall(d, p):
    """
    Return the unit l^p ball in R^d as a polytope.
    Only a polytope for p = 1, infty.
    """
    if p == 1:
        vertices = []
        for i in range(d):
            v = np.zeros(d)
            v[i] = 1
            w = np.zeros(d)
            w[i] = -1
            vertices.append(v)
            vertices.append(w)
        vertices = np.array(vertices)
    elif p == np.infty:
        vertices = 2 * np.array(all_subsets(d)) - 1
    else:
        raise NotImplementedError(f"l^p ball for p = {p} is not a polytope.")

    return Polytope(points=vertices)


def Cube(d, as_zonotope=False):
    """
    Return the unit cube in R^d. If as_zonotope=True,
    return as a Zonotope object rather than a Polytope.
    """
    if as_zonotope:
        generators = np.eye(d)
        return Zonotope(generators=generators)
    else:
        vertices = np.array(all_subsets(d))
        return Polytope(points=vertices)

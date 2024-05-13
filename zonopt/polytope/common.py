import numpy as np
from zonopt.polytope import Polytope, Zonotope
from zonopt.utils import all_subsets
import torch


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


def Cube(d, as_zonotope=False, use_torch=False):
    """
    Return the unit cube in R^d.

    Parameters:
    -----------
    as_zonotope: bool
        Return result as a zonotope.
    torch: bool
        If returning as a zonotope, return
        a zonotope with torch generators.
    """
    if as_zonotope:
        generators = np.eye(d)
        if use_torch:
            generators = torch.tensor(generators)
        return Zonotope(generators=generators)
    else:
        vertices = np.array(all_subsets(d))
        return Polytope(points=vertices)

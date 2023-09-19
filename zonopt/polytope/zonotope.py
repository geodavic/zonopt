import numpy as np
from typing import Union, List
from scipy.spatial import ConvexHull
from zonopt.polytope import Polytope
from zonopt.polytope.utils import (
    affine_from_vertices,
    vertices_from_affine,
)


class Zonotope(Polytope):
    """
    A zonotope in V representation or in generator-translation form.

    Parameters
    ----------
    generators: np.ndarray
        The generators of the zonotope.
    translation: np.ndarray
        The translation of the zonotope.
    points: np.ndarray
        A set of point whose convex hull is a zonotope.
    """

    def __init__(
        self,
        generators: Union[np.ndarray, List[List[float]]] = None,
        translation: Union[np.ndarray, List[float]] = None,
        points: Union[np.ndarray, List[List[float]]] = None,
    ):
        self.translation = translation
        if generators is not None:
            if self.translation is None:
                self.translation = np.zeros(len(generators[0]))
            self.generators = np.array(generators)
            _vert = vertices_from_affine(self.generators, self.translation)
        elif points is not None:
            _vert = Polytope(points=points).vertices
            self.generators, self.translation = affine_from_vertices(_vert)
        else:
            raise ValueError(
                "Must pass either points or generators that form the Zonotope"
            )

        hull = ConvexHull(_vert)
        super().__init__(hull=hull)

    def __mul__(self, a):
        """Override super class"""
        new_generators = []
        for g in self.generators:
            new_generators.append(a * g)
        return self.__class__(generators=new_generators)

    def __rmul__(self, a):
        """Same as __mul__"""
        return self.__mul__(a)

    @property
    def rank(self):
        return len(self.generators)

    @classmethod
    def random(
        self,
        rank: int,
        dimension: int,
        scale=None,
        positive=True,
        random_translation=False,
    ):
        """
        Return zonotope with randomly generated generators and translation.

        Parameters
        ----------
        rank: int
            rank of zonotope.
        dimension: int
            ambient dimension of zonotope.
        scale: int
            generators and translation are chosen with entries uniform random in
            [0,scale] or [-scale,scale], depending on the value of `positive`.
        positive: bool
            if true, the resulting zonotope will have all non-negative generators.
        """

        if scale is None:
            scale = np.sqrt(1 / rank)

        generators = np.random.rand(rank, dimension)
        if not positive:
            generators = 2 * generators - 1

        translation = np.zeros(dimension)
        if random_translation:
            translation = 2 * np.random.rand(dimension) - 1

        generators *= scale
        translation *= scale

        return self(generators=generators, translation=translation)

import numpy as np
from typing import Union, List, Any
import torch
from scipy.spatial import ConvexHull
from zonopt.polytope import Polytope
from zonopt.polytope.utils import hull_from_affine, translate_points

GeneratorType = Union[np.ndarray, torch.Tensor, List[List[float]]]
TranslationType = Union[np.ndarray, torch.Tensor, List[float]]


class Zonotope(Polytope):
    """
    A zonotope in V representation or in generator-translation form.

    Parameters
    ----------
    generators: np.ndarray
        The generators of the zonotope.
    translation: np.ndarray
        The translation of the zonotope.
    """

    def __init__(self, generators: GeneratorType, translation: TranslationType = None):
        if translation is None:
            translation = np.zeros(len(generators[0]))
            if isinstance(generators, torch.Tensor):
                translation = torch.tensor(translation)

        self.validate_generators(generators)
        self.validate_translation(translation)

        self._generators = generators
        self._translation = translation
        hull = hull_from_affine(generators, translation)

        super().__init__(hull=hull)

    def validate_generators(self, generators: Any):
        validationError = ValueError(f"Must pass type `{GeneratorType}` for generators")
        if isinstance(generators, (torch.Tensor, np.ndarray)):
            if len(generators.shape) != 2:
                raise ValueError(
                    "Generators must be specified as a two dimensional array"
                )
        elif not (
            isinstance(generators, List)
            and isinstance(generators[0], List)
            and isinstance(generators[0][0], float)
        ):
            raise validationError

    def validate_translation(self, translation: Any):
        validationError = ValueError(
            f"Must pass type `{TranslationType}` for translation"
        )
        if isinstance(translation, (torch.Tensor, np.ndarray)):
            if len(translation.shape) != 1:
                raise ValueError(
                    "Translations must be specified as a one dimensional array"
                )
        elif not (isinstance(translation, List) and isinstance(translation[0], float)):
            raise validationError

    @property
    def translation(self):
        return self._translation

    @property
    def generators(self):
        return self._generators

    @generators.setter
    def generators(self, new_generators: GeneratorType):
        self.validate_generators(new_generators)
        self._generators = new_generators
        self._hull = hull_from_affine(new_generators, self.translation)

    @translation.setter
    def translation(self, new_translation: TranslationType):
        self.validate_translation(new_translation)
        self._translation = new_translation
        _vert = translate_points(self.vertices, new_translation)
        self._hull = ConvexHull(_vert)

    @property
    def rank(self):
        return len(self.generators)

    @classmethod
    def random(
        cls,
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

        return cls(generators=generators, translation=translation)

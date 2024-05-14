import numpy as np
from typing import Union, List, Any
import torch
from scipy.spatial import ConvexHull
from zonopt.polytope import Polytope
from zonopt.polytope.utils import hull_from_affine, translate_points
from zonopt.polytope.comparisons import almost_equal

GeneratorType = Union[np.ndarray, torch.Tensor, List[List[float]]]
TranslationType = Union[np.ndarray, torch.Tensor, List[float]]


def express_as_subset_sum(
    x: np.ndarray, generators: List[np.ndarray], base: np.ndarray, return_indices=False
):
    """
    Express x as a sum of a subset of generators. Returns
    boolean and binary list corresponding to which generators
    add up to x.
    """
    if type(x) != type(generators[0]):
        raise ValueError("x and generators must be of the same array type")

    is_sum, gens = _express_as_subset_sum(x, generators, base)
    if return_indices:
        return is_sum, gens
    return is_sum, [i in gens for i in range(len(generators))]


def _express_as_subset_sum(
    x: np.ndarray, generators: List[np.ndarray], base: np.ndarray, _prev: int = 0
):
    """
    Helper function for above.
    """
    if almost_equal(x, base):
        return True, []
    if not len(generators):
        return False, []

    is_sum, subset = _express_as_subset_sum(
        x - generators[0], generators[1:], base, _prev=_prev + 1
    )
    if is_sum:
        return True, subset + [_prev]
    else:
        return _express_as_subset_sum(x, generators[1:], base, _prev=_prev + 1)


class Zonotope(Polytope):
    """
    A zonotope in generator-translation form.

    Parameters
    ----------
    generators: GeneratorType
        The generators of the zonotope.
    translation: TranslationType
        The translation of the zonotope.
    """

    def __init__(self, generators: GeneratorType, translation: TranslationType = None):
        if translation is None:
            translation = np.zeros(len(generators[0]))
            if isinstance(generators, torch.Tensor):
                translation = torch.tensor(translation)

        self._generators = self.validate_generators(generators)
        self._translation = self.validate_translation(translation)

        hull = hull_from_affine(self.generators, self.translation)
        super().__init__(hull=hull)

    def validate_generators(self, generators: Any):
        validationError = ValueError(f"Must pass type `{GeneratorType}` for generators")
        shapeError = ValueError(
            "Generators must be specified as a two dimensional array"
        )
        if isinstance(generators, np.ndarray):
            if len(generators.shape) != 2:
                raise shapeError
            return np.float64(generators)
        if isinstance(generators, torch.Tensor):
            if len(generators.shape) != 2:
                raise shapeError
            return generators
        elif (
            isinstance(generators, List)
            and isinstance(generators[0], List)
            and isinstance(generators[0][0], float)
        ):
            return np.float64(generators)
        else:
            raise validationError

    def validate_translation(self, translation: Any):
        validationError = ValueError(
            f"Must pass type `{TranslationType}` for translation"
        )
        shapeError = ValueError(
            "Translation must be specified as a one dimensional array."
        )
        if isinstance(translation, np.ndarray):
            if len(translation.shape) != 1:
                raise shapeError
            return np.float64(translation)
        if isinstance(translation, torch.Tensor):
            if len(translation.shape) != 1:
                raise shapeError
            return translation
        elif isinstance(translation, List) and isinstance(translation[0], float):
            return np.float64(translation)
        else:
            raise validationError

    @property
    def translation(self):
        return self._translation

    @property
    def generators(self):
        return self._generators

    @generators.setter
    def generators(self, new_generators: GeneratorType):
        self._generators = self.validate_generators(new_generators)
        self._hull = hull_from_affine(new_generators, self.translation)

    @translation.setter
    def translation(self, new_translation: TranslationType):
        self._translation = self.validate_translation(new_translation)
        _vert = translate_points(self.vertices, new_translation)
        self._hull = ConvexHull(_vert)

    @property
    def rank(self):
        return len(self.generators)

    def copy(self, requires_grad: bool = False):
        if isinstance(self.generators, torch.Tensor):
            generators_c = torch.clone(self.generators.detach())
            generators_c.requires_grad = requires_grad
        else:
            generators_c = np.copy(self.generators)

        if isinstance(self.translation, torch.Tensor):
            translation_c = torch.clone(self.translation.detach())
            translation_c.requires_grad = requires_grad
        else:
            translation_c = np.copy(self.translation)

        return self.__class__(generators_c, translation_c)

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

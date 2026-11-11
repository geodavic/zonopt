import numpy as np
from typing import Union, List, Any
import torch
import warnings
from scipy.spatial import ConvexHull
from scipy.optimize import lsq_linear
from zonopt.polytope import Polytope
from zonopt.polytope.utils import hull_from_affine, translate_points
from zonopt.polytope.comparisons import almost_equal
from zonopt.polytope.exceptions import GeometryError
from zonopt import global_config as config

GeneratorType = Union[np.ndarray, torch.Tensor, List[List[float]]]
TranslationType = Union[np.ndarray, torch.Tensor, List[float]]


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

        self._is_torch = False
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
            self._is_torch = True
            return generators
        elif (
            isinstance(generators, List)
            and isinstance(generators[0], List)
            and (
                isinstance(generators[0][0], float) or isinstance(generators[0][0], int)
            )
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
        elif isinstance(translation, List) and (
            isinstance(translation[0], float) or isinstance(translation[0], int)
        ):
            return np.float64(translation)
        else:
            raise validationError

    @property
    def translation(self):
        return self._translation

    @property
    def generators(self):
        return self._generators

    @property
    def is_torch(self):
        return self._is_torch

    @generators.setter
    def generators(self, new_generators: GeneratorType):
        self._generators = self.validate_generators(new_generators)
        self._hull = hull_from_affine(new_generators, self.translation)

    @translation.setter
    def translation(self, new_translation: TranslationType):
        self._translation = self.validate_translation(new_translation)
        self._hull = hull_from_affine(self.generators, self._translation)

    @property
    def rank(self):
        return len(self.generators)

    def copy(self, cast_to_torch: bool = False, requires_grad: bool = False):
        if isinstance(self.generators, torch.Tensor):
            generators_c = torch.clone(self.generators.detach())
            generators_c.requires_grad = requires_grad
        elif cast_to_torch:
            generators_c = torch.tensor(self.generators, requires_grad=requires_grad)
        else:
            generators_c = np.copy(self.generators)

        if isinstance(self.translation, torch.Tensor):
            translation_c = torch.clone(self.translation.detach())
            translation_c.requires_grad = requires_grad
        elif cast_to_torch:
            translation_c = torch.tensor(self.translation, requires_grad=requires_grad)
        else:
            translation_c = np.copy(self.translation)

        return self.__class__(generators_c, translation_c)

    def lift(self, x: np.ndarray, tol: float = 1e-6):
        """
        Lift a point x of Z to the unit cube: find epsilon in [0,1]^rank with
        Q^T epsilon + mu = x, where Q is the generator matrix and mu the
        translation. If x lies on the boundary of a stable (e.g. general
        position) zonotope, this point is unique. For x in the interior the
        lift is not unique and one solution is returned.

        Solved as a bounded least squares problem. Raises GeometryError if
        the residual exceeds `tol`, i.e. x is not (numerically) in Z.

        Parameters
        ----------
        x: np.ndarray
            A point in Z.
        tol: float
            Maximum allowed residual ||Q^T epsilon + mu - x||.

        Returns
        -------
        epsilon: np.ndarray
            Coordinates in the unit cube, with entries within
            `comparison_epsilon` of 0 or 1 snapped to those values.
        """
        if self.is_torch:
            generators = self.generators.detach().numpy()
            translation = self.translation.detach().numpy()
        else:
            generators = self.generators
            translation = self.translation
        x = np.asarray(x, dtype=np.float64)

        result = lsq_linear(
            generators.T, x - translation, bounds=(0, 1), method="bvls", tol=1e-14
        )
        epsilon = result.x
        residual = np.linalg.norm(generators.T @ epsilon + translation - x)
        if residual > tol:
            raise GeometryError(
                f"Point {x} is not in the zonotope (lift residual {residual:.2e})."
            )

        snap = config.comparison_epsilon
        epsilon[np.abs(epsilon) < snap] = 0.0
        epsilon[np.abs(epsilon - 1) < snap] = 1.0
        return epsilon

    @classmethod
    def random(
        cls,
        rank: int,
        dimension: int,
        scale: float = None,
        seed: int = None,
        positive: bool = True,
        random_translation: bool = False,
        use_torch: bool = False,
        requires_grad: bool = False,
    ):
        """
        Return zonotope with randomly generated generators and translation.

        Parameters
        ----------
        rank: int
            rank of zonotope.
        dimension: int
            ambient dimension of zonotope.
        scale: float
            generators and translation are chosen with entries uniform random in
            [0,scale] or [-scale,scale], depending on the value of `positive`.
        seed: int
            Seed used for the np random call.
        positive: bool
            if true, the resulting zonotope will have all non-negative generators.
        random_translation: bool
            Whether to use a random translation instead of zero translation.
        use_torch: bool
            Return a torch zonotope (tensor parameters)
        requires_grad: bool
            If above is true, then make tensors require grad.
        """

        if seed is not None:
            np.random.seed(seed)

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

        if use_torch:
            generators = torch.tensor(generators, requires_grad=requires_grad)
            translation = torch.tensor(translation, requires_grad=requires_grad)

        return cls(generators=generators, translation=translation)


def express_as_subset_sum(x: np.ndarray, Z: Zonotope, return_indices=False):
    """
    Express x as a sum of a subset of the generators of Z. Returns
    boolean and a list of indices of generators of Z which are used
    in the sum.
    """

    if Z.is_torch:
        generators = list(Z.generators.detach().numpy())
        base = Z.translation.detach().numpy()
    else:
        generators = list(Z.generators)
        base = Z.translation

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
    Recursive helper function for above.
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

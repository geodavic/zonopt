import numpy as np
from typing import Union, List, Any
import torch
import warnings
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.optimize import lsq_linear
from zonopt.polytope import Polytope
from zonopt.polytope.utils import hull_from_affine, translate_points, binary_lists
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
        self._invalidate_geometry()

    # ------------------------------------------------------------------
    # Geometry (vertices and facets), computed lazily and cached.
    # ------------------------------------------------------------------

    def _invalidate_geometry(self):
        self._vertices_cache = None
        self._equations_cache = None
        self._general_position = None

    def _numpy_parameters(self):
        if self.is_torch:
            return (
                self.generators.detach().numpy().astype(np.float64),
                self.translation.detach().numpy().astype(np.float64),
            )
        return self.generators, self.translation

    def _ensure_geometry(self):
        if self._vertices_cache is None:
            generators, translation = self._numpy_parameters()
            V, E, gp = zonotope_geometry(generators, translation)
            self._vertices_cache = V
            self._equations_cache = E
            self._general_position = gp

    @property
    def points(self):
        return self.vertices

    @property
    def vertices(self):
        self._ensure_geometry()
        return self._vertices_cache

    @property
    def equations(self):
        self._ensure_geometry()
        return self._equations_cache

    @property
    def is_general_position(self):
        """
        Whether the generators were found to be in general position by the
        facet enumeration (every d-1 of them independent and no further
        generator parallel to a facet).
        """
        self._ensure_geometry()
        return self._general_position

    @property
    def dimension(self):
        return int(self.generators.shape[1])

    @property
    def barycenter(self):
        generators, translation = self._numpy_parameters()
        return translation + 0.5 * generators.sum(axis=0)

    def __mul__(self, alpha):
        return self.__class__(alpha * self.generators, alpha * self.translation)

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
        self._invalidate_geometry()

    @translation.setter
    def translation(self, new_translation: TranslationType):
        self._translation = self.validate_translation(new_translation)
        self._invalidate_geometry()

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

        new = self.__class__(generators_c, translation_c)
        # Same geometry: share the cached vertices and facets rather than recomputing.
        new._vertices_cache = self._vertices_cache
        new._equations_cache = self._equations_cache
        new._general_position = self._general_position
        return new

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


def _unique_rows(E: np.ndarray):
    """
    Unique rows of a boolean matrix. Rows are packed into integers so that the
    deduplication is a 1-d unique, which is far faster than np.unique(axis=0).
    """
    n = E.shape[1]
    if n > 62:
        return np.unique(E, axis=0)
    codes = E.astype(np.int64) @ (np.int64(1) << np.arange(n, dtype=np.int64))
    _, index = np.unique(codes, return_index=True)
    # Rows come back ordered by code, i.e. by cube coordinates, so the origin
    # vertex (if present) is first and the all-generators vertex is last.
    return E[index]


def zonotope_geometry(generators: np.ndarray, translation: np.ndarray, tol: float = None):
    """
    Vertices and facet inequalities of the zonotope Z = { Q^T e + mu : e in [0,1]^n }
    without enumerating the 2^n cubical vertices.

    Every facet of a zonotope is a translate of the zonotope spanned by the
    generators parallel to it. In general position those are exactly d-1
    generators S, the facet normal w is the null vector of Q_S (computed here
    from the (d-1)x(d-1) minors, w_j = (-1)^j det(Q_S without column j)), and
    the two facets with normals +w and -w have vertices

        mu + sum_{<g_i, w> > 0} g_i + (subset sum of S)   and
        mu + sum_{<g_i, w> < 0} g_i + (subset sum of S).

    Every vertex lies on a facet, so the union over all (d-1)-subsets is the
    vertex set. Cost is O(C(n, d-1) 2^(d-1)) instead of O(2^n).

    If the generators are not in general position (some d-1 of them dependent,
    or a further generator parallel to a facet), the candidate points above
    may include non-vertices and miss vertices on facets with more than d-1
    parallel generators, so we fall back to a convex hull of an enlarged
    candidate set.

    Returns
    -------
    vertices: np.ndarray
        Vertices of Z, ordered by their cube coordinates (the translation
        vertex first, the sum of all generators last, when these are
        vertices). In dimension 2 they are instead in counterclockwise order.
    equations: np.ndarray
        Rows [a, -c] with a a unit outward normal and a . x - c <= 0 on Z.
    general_position: bool
    """
    if tol is None:
        tol = config.comparison_epsilon
    generators = np.asarray(generators, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    n, d = generators.shape

    if d < 2 or n < d - 1:
        hull = hull_from_affine(generators, translation)
        return (
            hull.points[hull.vertices],
            np.unique(hull.equations, axis=0),
            False,
        )

    subsets = np.array(list(combinations(range(n), d - 1)), dtype=int).reshape(
        -1, d - 1
    )
    Q_S = generators[subsets]  # (C, d-1, d)

    # Facet normals from the minors of Q_S (cf. Proposition grad2 of the paper).
    W = np.empty((len(subsets), d))
    for j in range(d):
        cols = [k for k in range(d) if k != j]
        W[:, j] = (-1) ** j * np.linalg.det(Q_S[:, :, cols])
    norms = np.linalg.norm(W, axis=1)
    scale = np.max(np.linalg.norm(generators, axis=1)) ** (d - 1)
    independent = norms > tol * max(scale, 1e-300)
    if not independent.any():
        raise GeometryError("Zonotope is not full dimensional.")
    general_position = bool(independent.all())
    W = W[independent] / norms[independent, None]
    subsets = subsets[independent]
    C = len(subsets)

    dots = W @ generators.T  # (C, n)
    in_S = np.zeros((C, n), dtype=bool)
    in_S[np.arange(C)[:, None], subsets] = True
    positive = (dots > tol) & ~in_S
    negative = (dots < -tol) & ~in_S
    parallel = ~positive & ~negative & ~in_S
    general_position = general_position and not parallel.any()

    # Cube coordinates of the facet vertices: positive side, plus every subset of S.
    patterns = np.array(binary_lists(d - 1), dtype=bool).reshape(-1, d - 1)
    m = len(patterns)
    facet_of_row = np.repeat(np.arange(C), m)  # which facet each candidate row belongs to
    row_index = np.arange(C * m)[:, None]
    columns = subsets[facet_of_row]  # generator indices of S for that facet
    E_plus = np.repeat(positive, m, axis=0)
    E_plus[row_index, columns] = np.tile(patterns, (C, 1))
    E_minus = np.repeat(negative, m, axis=0)
    E_minus[row_index, columns] = np.tile(patterns, (C, 1))
    E = _unique_rows(np.vstack([E_plus, E_minus]))

    if general_position:
        vertices = E.astype(np.float64) @ generators + translation
        base_plus = translation + positive.astype(np.float64) @ generators
        base_minus = translation + negative.astype(np.float64) @ generators
        c_plus = np.einsum("cd,cd->c", W, base_plus)
        c_minus = np.einsum("cd,cd->c", -W, base_minus)
        equations = np.vstack(
            [
                np.hstack([W, -c_plus[:, None]]),
                np.hstack([-W, -c_minus[:, None]]),
            ]
        )
    else:
        # Enlarge the candidate set with all subset sums of S together with the
        # generators parallel to the facet, then let Qhull pick out the vertices.
        extra = []
        for c in np.flatnonzero(parallel.any(axis=1)):
            T = np.flatnonzero(in_S[c] | parallel[c])
            pats = np.array(binary_lists(len(T)), dtype=bool).reshape(-1, len(T))
            for base in (positive[c], negative[c]):
                rows_T = np.repeat(base[None, :], len(pats), axis=0)
                rows_T[:, T] = pats
                extra.append(rows_T)
        if extra:
            E = _unique_rows(np.vstack([E] + extra))
        candidates = E.astype(np.float64) @ generators + translation
        hull = ConvexHull(candidates)
        vertices = candidates[hull.vertices]
        equations = np.unique(hull.equations, axis=0)

    if d == 2:
        center = translation + 0.5 * generators.sum(axis=0)
        angles = np.arctan2(*(vertices - center).T[::-1])
        vertices = vertices[np.argsort(angles)]

    return vertices, equations, general_position

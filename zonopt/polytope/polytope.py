import numpy as np
from typing import List, Union
from scipy.spatial import ConvexHull
from zonopt.polytope.comparisons import almost_equal
from zonopt.polytope.utils import is_centrally_symmetric, express_point_as_convex_sum
from zonopt.polytope.exceptions import GeometryError
from zonopt.config import global_config as config


class Hyperplane:
    """
    A hyperplane of the form a \odot x - c == 0
    """

    def __init__(self, a, c):
        self.a = np.array(a)
        self.c = c
        self._a = self.a / np.linalg.norm(a)
        self._c = self.c / np.linalg.norm(a)

    def __eq__(self, other):
        if isinstance(other, Hyperplane):
            e1 = almost_equal(self._a, other._a) and almost_equal(self._c, other._c)
            e2 = almost_equal(self._a, -other._a) and almost_equal(self._c, -other._c)
            return e1 or e2
        return False

    def contains(self, x):
        if almost_equal(self._a @ x, self._c):
            return True
        return False

    def __repr__(self):
        a = np.round(self.a, 3)
        c = np.round(self.c, 3)
        return f"Hyperplane({a}•x = {c})"


class Halfspace:
    """
    A halfspace of the form a \dot x - c <= 0
    """

    def __init__(self, a, c):
        self.boundary = Hyperplane(a, c)
        self.a = np.array(a)
        self.c = c
        self._a = self.a / np.linalg.norm(a)
        self._c = self.c / np.linalg.norm(a)

    def __eq__(self, other):
        if isinstance(other, Halfspace):
            e1 = almost_equal(self._a, other._a) and almost_equal(self._c, other._c)
            e2 = almost_equal(self._a, -other._a) and almost_equal(self._c, -other._c)
            return e1 or e2
        return False

    def contains(self, x):
        if self.boundary.contains(x):
            return True
        if self._a @ x - self._c <= 0:
            return True
        return False

    def __repr__(self):
        a = np.round(self.a, 3)
        c = np.round(self.c, 3)
        return f"Halfspace({a}•x =< {c})"


class Polytope:
    """
    A polytope, stored by its vertices and its facet inequalities.

    Parameters
    ----------
    hull: scipy.spatial.ConvexHull
        A hull object
    points: Union[np.ndarray, List[float]]
        A list of points from which to form the convex hull.
    """

    def __init__(
        self,
        hull: ConvexHull = None,
        points: Union[np.ndarray, List[List[float]]] = None,
    ):
        if hull is None and points is None:
            raise TypeError(
                "Must specify either 'hull' or 'points' when initializing a Polytope."
            )
        if hull is None:
            hull = ConvexHull(points)
        self._set_from_hull(hull)

    def _set_from_hull(self, hull: ConvexHull):
        self._points = np.asarray(hull.points, dtype=np.float64)
        self._vertices = self._points[hull.vertices]
        self._equations = np.unique(hull.equations, axis=0)

    @classmethod
    def random(self, num_points: int, dimension: int, scale=1, seed: int = None):
        """
        Return polytope formed by the convex hull of a given number of random points
        in the given dimension.
        """
        if seed is not None:
            np.random.seed(seed)
        pts = scale * np.random.rand(num_points, dimension)
        P = self(points=pts)
        return P

    @property
    def points(self):
        return self._points

    @property
    def vertices(self):
        return self._vertices

    @property
    def equations(self):
        """
        Facet inequalities as an array with rows [a, -c], where a is a unit
        outward normal and the facet inequality is a . x - c <= 0 (same
        convention as scipy.spatial.ConvexHull.equations).
        """
        return self._equations

    @property
    def halfspaces(self):
        return [Halfspace(eq[:-1], -eq[-1]) for eq in self.equations]

    def __mul__(self, alpha):
        """
        Multiply by scalar
        """
        new_points = []
        for v in self.points:
            new_points.append(alpha * v)
        return self.__class__(points=new_points)

    def __rmul__(self, alpha):
        """
        Same as __mul__
        """
        return self.__mul__(alpha)

    @property
    def dimension(self):
        return self.vertices.shape[1]

    @property
    def barycenter(self):
        """
        Get barycenter of polytope
        """
        return np.sum(self.vertices, axis=0) / len(self.vertices)

    def is_centrally_symmetric(self):
        return is_centrally_symmetric(self.vertices)

    def has_vertex(self, x: np.ndarray):
        """
        Return true if x is (almost) a vertex of P.
        """
        x = np.asarray(x, dtype=np.float64)
        distances = np.linalg.norm(self.vertices - x, axis=1)
        return bool(np.any(distances < config.comparison_epsilon))

    def _facet_values(self, x: np.ndarray):
        """
        The values a_i . x - c_i for every facet inequality; negative inside,
        zero on the facet, positive outside.
        """
        x = np.asarray(x, dtype=np.float64)
        return self.equations[:, :-1] @ x + self.equations[:, -1]

    def contains(self, p: np.ndarray):
        """
        Determine if the polytope contains p (up to comparison_epsilon).
        """
        return bool(np.all(self._facet_values(p) < config.comparison_epsilon))

    def supporting_halfspaces(self, x):
        """
        Return list of halfspaces whose boundary contains x
        """
        mask = np.abs(self._facet_values(x)) < config.comparison_epsilon
        return [Halfspace(eq[:-1], -eq[-1]) for eq in self.equations[mask]]

    def face_dimension(self, x):
        """
        Return the dimension of the face on which x lies. If
        x is not in P, then return -1.
        """
        if not self.contains(x):
            return -1

        incident = self.equations[
            np.abs(self._facet_values(x)) < config.comparison_epsilon
        ]
        # Vertices lying on every incident facet are the vertices of the face
        values = self.vertices @ incident[:, :-1].T + incident[:, -1]
        active_vertices = self.vertices[
            np.all(np.abs(values) < config.comparison_epsilon, axis=1)
        ]

        barycenter = active_vertices.mean(axis=0)

        # Translate to barycenter so that span of active vertices is a subspace.
        # Use an explicit tolerance: numpy's default is of order machine epsilon,
        # which rounding in `v - barycenter` can exceed (e.g. two antipodal
        # points of an edge coming out as rank 2).
        return int(
            np.linalg.matrix_rank(
                active_vertices - barycenter, tol=config.comparison_epsilon
            )
        )

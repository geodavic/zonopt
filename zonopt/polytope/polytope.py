import numpy as np
from typing import List, Union
from scipy.spatial import ConvexHull
from zonopt.polytope.comparisons import almost_equal
from zonopt.polytope.utils import is_centrally_symmetric, express_point_as_convex_sum
from zonopt.polytope.exceptions import GeometryError


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
    A polytope.

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
            self._hull = ConvexHull(points)
        else:
            self._hull = hull

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
        return self._hull.points

    @property
    def vertices(self):
        return self.points[self._hull.vertices]

    @property
    def halfspaces(self):
        inequalities = [e for e in np.unique(self._hull.equations, axis=0)]
        return [Halfspace(eq[:-1], -eq[-1]) for eq in inequalities]

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
        return len(self.points[0])

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
        return any([almost_equal(np.linalg.norm(x - v), 0) for v in self.vertices])

    def contains(self, p: np.ndarray):
        """
        Determine if the polytope contains p
        """
        for H in self.halfspaces:
            if not H.contains(p):
                return False
        return True

    def supporting_halfspaces(self, x):
        """
        Return list of halfspaces whose boundary contains x
        """
        halfspaces = []
        for H in self.halfspaces:
            if H.boundary.contains(x):
                halfspaces.append(H)
        return halfspaces

    def face_dimension(self, x):
        """
        Return the dimension of the face on which x lies. If
        x is not in P, then return -1.
        """
        if not self.contains(x):
            return -1

        incident_h = self.supporting_halfspaces(x)
        active_vertices = []
        for v in self.vertices:
            if all([H.boundary.contains(v) for H in incident_h]):
                active_vertices.append(v)

        barycenter = sum(active_vertices) / len(active_vertices)

        # Translate to barycenter so that span of active vertices is a subspace
        active_vertices_translated = [v - barycenter for v in active_vertices]
        
        return np.linalg.matrix_rank(active_vertices_translated)

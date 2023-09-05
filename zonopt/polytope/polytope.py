import numpy as np
from typing import List, Union
from scipy.spatial import ConvexHull
from .tolerance import near

class Hyperplane:
    """ A hyperplane of the form a \odot x - c == 0 """
    
    def __init__(self, a, c):
        self.a = np.array(a)
        self.c = c

    def __eq__(self, other):
        if isinstance(other, Hyperplane):
            return near(self.a, other.a) and near(self.c, other.c)
        return False

    def contains(self, x):
        if near(self.a @ x, self.c):
            return True
        return False


class Halfspace:
    """A halfspace of the form a \dot x - c <= 0"""

    def __init__(self, a, c):
        self.boundary = Hyperplane(a,c)
        self.a = np.array(a)
        self.c = c

    def __eq__(self, other):
        if isinstance(other, Halfspace):
            return near(self.a, other.a) and near(self.c, other.c)
        return False

    def contains(self, x):
        if self.a @ x - self.c <= 0:
            return True
        return False


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

    def __init__(self, 
        hull: ConvexHull = None,
        points: Union[np.ndarray, List[float]] = None
    ):
        if hull is None and points is None:
            raise TypeError("Must specify either 'hull' or 'points' when initializing a Polytope.")
        if hull is None:
            self._hull = ConvexHull(points)
        else:
            self._hull = hull

    @peroperty
    def points(self):
        return self._hull.points

    @property
    def vertices(self):
        return self.points[self._hull.vertices]

    @property
    def halfspaces(self):
        inequalities = [e for e in np.unique(self._hull.equations, axis = 0)]
        return [Halfspace(eq[:-1], -eq[-1]) for eq in inequalities]

    def __mul__(self, alpha):
        """ Multiply by scalar """
        new_points = []
        for v in self.points:
            new_points.append(alpha* v)
        return self.__class__(points=new_points)

    def __rmul__(self, alpha):
        """ Same as __mul__ """
        return self.__mul__(alpha)

    @property
    def dimension(self):
        return len(self.points[0])

    @property
    def barycenter(self):
        """ Get barycenter of polytope """
        return np.sum(self.vertices, axis = 0) / len(self.vertices)

    def contains(self, p: np.ndarray):
        """ Determine if the polytope contains p """
        for H in self.halfspaces:
            if not H.contains(p):
                return False
        return True

    def supporting_halfspaces(self, x):
        """ Return list of halfspaces whose boundary contains x """
        halfspaces = []
        for H in self.halfspaces:
            if H.boundary.contains(x):
                halfspaces.append(H)
        return halfspaces

import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from zonopt.polytope import Polytope
from zonopt.plot.utils import get_bounds


class Plotter:
    """
    Draw polytopes in two dimensions.
    """

    def __init__(self, figsize=None, pad=0.4, marker="o", verbose=True):
        self.figsize = figsize or (10, 10)
        self.pad = pad
        self.marker = marker
        self.verbose = verbose
        self._setup()

    def _setup(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_aspect("equal")

    def draw_polytopes(
        self,
        polytopes: Union[List[Polytope], Polytope],
        color: Union[List[str], str] = None,
        pin_first: bool = False,
        bounds: tuple = None,
        axes: bool = True,
    ):
        """
        Render a collection of Polytopes.

        Parameters:
        -----------
        polytopes: Union[List[Polytope],Polytope]
            The polytopes to be rendered
        pin_first: bool
            Whether to center plot and bounds on first polytope.
        bounds: tuple
            (min,max) pair for the x and y bounds.
        """

        if isinstance(polytopes, Polytope):
            polytopes = [polytopes]

        if color is None:
            color = "k"
        if isinstance(color, str):
            color = [color] * len(polytopes)

        if len(color) != len(polytopes):
            raise ValueError(
                "Must pass either a single color or the same number of colors as polytopes"
            )

        if bounds is None:
            maxP, minP = (
                get_bounds(polytopes[:1]) if pin_first else get_bounds(polytopes)
            )
            extra_width = self.pad * max(maxP[0] - minP[0], maxP[1] - minP[1])
            bounds = (min(minP) - extra_width, max(maxP) + extra_width)

        if not axes:
            self.ax.set_axis_off()

        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[0], bounds[1])

        for c, P in zip(color, polytopes):
            self._draw_single(P, c)

    def draw_lines(
        self,
        lines: List[Tuple],
        marker: Union[List[str], str] = None,
        color: Union[List[str], str] = None,
    ):
        """
        Render a collection of lines.

        Parameters:
        -----------
        lines: List[Tuple]
            List of lines, where each line is a tuple (p,q) of points in the plane.
        marker: Union[List[str], str]
            Marker(s) for the endpoints of the lines.
        color: Union[List[str], str]
            Color(s) for the lines.
        """
        if color is None:
            color = "k"
        if isinstance(color, str):
            color = [color] * len(lines)

        if marker is None:
            marker = "*"
        if isinstance(marker, str):
            marker = [marker] * len(lines)

        if len(marker) != len(lines):
            raise ValueError(
                "Marker must be a single string or a list of strings the same length as lines"
            )
        if len(color) != len(lines):
            raise ValueError(
                "Color must be a single string or a list of strings the same length as lines"
            )

        for line, m, c in zip(lines, marker, color):
            self._draw_line(line[0], line[1], marker=m, color=c)

    def _draw_single(self, P: Polytope, color: str):
        n = len(P.vertices)
        for i in range(n):
            p1 = P.vertices[i]
            p2 = P.vertices[(i + 1) % n]
            self.ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], marker=self.marker, color=color
            )

    def _draw_line(self, p, q, marker=None, color=None):
        self.ax.plot([p[0], q[0]], [p[1], q[1]], marker=marker, color=color)

    def save(self, filename=None, dpi=300):
        """
        Export to a file.
        """
        if filename is None:
            filename = "render.png"
        self.ax.get_figure().savefig(filename, bbox_inches="tight", dpi=dpi)
        if self.verbose:
            print(f"Saved as {filename}")
        self.fig.clf()
        self._setup()

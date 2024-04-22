from zonopt.metrics import hausdorff_distance
from zonopt.polytope import Polytope, Zonotope
import torch

def zonotope_loss(P: Polytope, generators: torch.Tensor, translation: torch.Tensor):
    """
    Hausdorff distance between P and a zonotope viewed as a loss
    function. Calculates the hausdorff distance between P and the
    zonotope in a differentiable way.
    """

    pass

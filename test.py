from zonopt.metrics import hausdorff_distance
from zonopt.polytope import Zonotope, Polytope
import torch
import numpy as np

generators = torch.eye(2, requires_grad=True, dtype=torch.float64)
translation = torch.zeros(2, requires_grad=True, dtype=torch.float64)
Z = Zonotope(generators=generators, translation=translation)
th = np.pi/4
rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
P = Polytope(points=np.array([rot@v for v in Z.vertices]+np.array([4,4-np.sqrt(2)/2])))
d, p, q = hausdorff_distance(P,Z)

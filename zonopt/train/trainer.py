from zonopt.polytope import Polytope, Zonotope
from zonopt.metrics import hausdorff_distance
from zonopt.train.loss import hausdorff_loss
from zonopt.train.optimizer import Optimizer
import numpy as np
import torch


class ZonotopeTrainer:
    def __init__(
        self,
        target_polytope: Polytope,
        optimizer: Optimizer,
        zonotope_rank: int = None,
        start: Zonotope = None,
        seed: int = None,
        warmstart: bool = False,
    ):
        self.target_polytope = target_polytope
        self.seed = seed or np.random.randint(2**32)
        self.warmstart = warmstart

        if start is None:
            if zonotope_rank is None:
                raise ValueError("Must pass a zonotope rank or a start zonotope")
            start = self._initialize_zonotope(zonotope_rank)

        if start.is_torch:
            raise ValueError("Starting zonotpe must be a numpy zonotope")

        self.zonotope = start.copy()
        self.optimizer.zonotope = self.zonotope

    def _initialize_zonotope(self, rank: int):
        """
        TODO
        """
        pass

    def _single_train_step(self, target_points: np.ndarray, control_points: np.ndarray):
        gradients_data = []
        for p, q in zip(target_points, control_points):
            Z = self.zonotope.copy(cast_to_torch=True, requires_grad=True)
            loss = hausdorff_loss(Z, self.target_polytope, q, p)
            loss.backward()
            gradients_data += [
                (
                    self._get_grad(Z.generators),
                    self._get_grad(Z.translation),
                    p,
                    q,
                    loss.item()
                )
            ]

        self.optimizer.step(gradients_data)

    def _get_grad(self, t: torch.Tensor):
        if t.grad is None:
            return np.zeroslike(t)
        return t.grad.detach().numpy()

    def train(self):
        """
        TODO
        """
        for step in pbar:
            distance, target_points, control_points = hausdorff_distance(
                self.target_polytope, self.zonotope
            )
            self._single_train_step(target_points, control_points)

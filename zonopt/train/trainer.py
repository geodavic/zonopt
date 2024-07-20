from zonopt.polytope import Polytope, Zonotope
from zonopt.metrics import hausdorff_distance
from zonopt.train.loss import hausdorff_loss
from zonopt.train.optimizer import Optimizer
from zonopt.todo import GeorgePleaseImplement
import numpy as np
import torch
from tqdm import tqdm


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
        print(self.seed)
        self.warmstart = warmstart
        self.optimizer = optimizer

        if start is None:
            if zonotope_rank is None:
                raise ValueError("Must pass a zonotope rank or a start zonotope")
            start = self._initialize_zonotope(zonotope_rank, target_polytope.dimension)

        if start.is_torch:
            raise ValueError("Starting zonotpe must be a numpy zonotope")

        self.optimizer.zonotope = start.copy()

    def _initialize_zonotope(self, rank: int, dimension: int):
        if self.warmstart:
            raise GeorgePleaseImplement("warmstarting")

        # TODO: add seed, scale, positive, random_translation options here
        return Zonotope.random(rank, dimension, seed=self.seed)

    def _single_train_step(self, target_points: np.ndarray, control_points: np.ndarray):
        gradients_data = []
        for p, q in zip(target_points, control_points):
            Z = self.optimizer.zonotope.copy(cast_to_torch=True, requires_grad=True)
            loss = hausdorff_loss(Z, self.target_polytope, q, p)
            loss.backward()

                pdb.set_trace()
            gradients_data += [
                (
                    self._get_grad(Z.generators),
                    self._get_grad(Z.translation),
                    p,
                    q,
                    loss.item(),
                )
            ]

        self.optimizer.step(gradients_data)

    def _get_grad(self, t: torch.Tensor):
        if t.grad is None:
            return np.zeros_like(t.detach().numpy())
        return t.grad.detach().numpy()

    def train(self, num_steps: int):
        """
        Main training loop.
        TODO: Add more kwargs
        """
        pbar = tqdm(range(num_steps))
        for step in pbar:
            distance, target_points, control_points = hausdorff_distance(
                self.target_polytope, self.optimizer.zonotope
            )
            self._single_train_step(target_points, control_points)
            pbar.set_description(f"d = {distance:10.10f}")
        return self.optimizer.zonotope

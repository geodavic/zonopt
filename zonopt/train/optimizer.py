from typing import List, Tuple
from zonopt.train.lrscheduler import LRScheduler, SimpleLRScheduler
from zonopt.train.subdifferential import feasible_subdifferential_center
from zonopt.polytope import Zonotope
import numpy as np


class Optimizer:
    def __init__(
        self,
        lr: float = None,
        lr_scheduler: LRScheduler = None,
        normalize_grad: bool = False,
        use_feasible_subdifferential: bool = False,
    ):
        if lr is None and lr_scheduler is None:
            raise ValueError("Must specify either an LRScheduler or a learning rate")
        self.lr_scheduler = lr_scheduler or SimpleLRScheduler(lr)
        if lr is not None:
            self.lr_scheduler.lr = lr
        self.normalize_grad = normalize_grad
        self.use_feasible_subdifferential = use_feasible_subdifferential
        self._zonotope = None

    @property
    def zonotope(self):
        return self._zonotope

    @zonotope.setter
    def zonotope(self, Z: Zonotope):
        if Z.is_torch:
            raise ValueError("Optimizer zonotope must be numpy")
        self._zonotope = Z

    def flatten_gradient(
        self, generator_grad: np.ndarray, translation_grad: np.ndarray
    ):
        return np.concatenate([generator_grad.flatten(), translation_grad])

    def apply_gradient(self, gradient: np.ndarray):
        if self.normalize_grad:
            gradient = gradient / np.linalg.norm(gradient)

        n = self.zonotope.dimension
        d = self.zonotope.rank
        generator_grad = gradient[: n * d].reshape(d, n)
        translation_grad = gradient[n * d :]

        self.zonotope.generators -= self.lr_scheduler.lr * generator_grad
        self.zonotope.translation -= self.lr_scheduler.lr * translation_grad

    def step(self, gradients_data: List[Tuple]):
        data = sorted(gradients_data, key=lambda x: x[-1])  # Sort by loss value

        if not self.use_feasible_subdifferential:
            # Step in only one direction
            self.apply_gradient(self.flatten_gradient(data[0][0], data[0][1]))
        else:
            # Step in the direction of the Chebyshev center of the feasible subdifferential
            grad = feasible_subdifferential_center(self.zonotope, data)
            self.apply_gradient(grad)

        self.lr_scheduler.step()

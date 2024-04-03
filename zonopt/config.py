from pydantic import BaseModel, field_validator, ValidationError
import math


class QPConfig(BaseModel):
    """
    Configuration for the quadratic program
    solver that computes the L^2 Hausdorff distance.

    For details on these paramters, see
    https://github.com/qpsolvers/qpsolvers
    """

    solver: str = "osqp"
    polish: int = 1
    max_iter: int = 4000000
    eps_abs: float = 1e-11
    eps_rel: float = 1e-11


class OptimizationConfig(BaseModel):
    """
    Configuration for zonopt.

    Parameters
    ----------
    metric: int
        the L^p metric under which the distance between the
        target polytope and a zonotope is optimized.
    comparison_epsilon: float
        Threshold used to compare array-likes. Two
        array-likes whose distance (under comparison_metric) is
        less than this value are considered equal.
    comparison_metric: int
        L^p metric under which array-likes are compared.
    """

    metric: int = 2
    comparison_epsilon: float = 1e-8
    comparison_metric: int = 2
    qp_config: QPConfig = QPConfig()

    @field_validator("metric", "comparison_metric")
    def must_be_p_metric(cls, v, info):
        if v < 1:
            raise ValueError(
                f"`{info.field_name}` must be a p-metric with 1 <= p <= np.infty"
            )
        return v

    @field_validator("comparison_epsilon")
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Comparison epsilon must be > 0")
        return v

    @property
    def log_comparison_epsilon(self):
        return -int(math.log10(self.comparison_epsilon))


global_config = OptimizationConfig()

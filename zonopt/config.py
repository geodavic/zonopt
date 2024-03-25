from pydantic import BaseModel, field_validator, ValidationError
import math

class OptimizationConfig(BaseModel):
    """
    Configuration for zonopt.

    Parameters
    ----------
    metric: the default p-metric used 
    """
    metric: int = 2
    comparison_epsilon: float = 1e-8
    comparison_metric: int = 2

    @field_validator("metric","comparison_metric")
    def must_be_p_metric(cls, v, info):
        if v < 1:
            raise ValueError(f"`{info.field_name}` must be a p-metric with 1 <= p <= np.infty")
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

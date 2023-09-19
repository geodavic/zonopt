import numpy as np
import math

_global_epsilon = 1e-8
_comparison_norm = 2
_log_global_epsilon = -int(math.log10(_global_epsilon))


def almost_equal(o1: any, o2: any):
    """
    Check if two things are almost equal.
    """

    if type(o1) != type(o2):
        raise TypeError(f"Cannot compare objects of type '{type(o1)}' and '{type(o2)}'")

    if type(o1) == float:
        return np.abs(o1 - o2) < _global_epsilon
    elif type(o1) == np.ndarray:
        return np.linalg.norm(o1 - o2, _comparison_norm) < _global_epsilon
    else:
        raise TypeError(f"Unsupported type '{type(o1)}' for comparison")

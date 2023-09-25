import numpy as np
import math

_global_epsilon = 1e-8
_comparison_norm = 2
_log_global_epsilon = -int(math.log10(_global_epsilon))


def almost_equal(o1: any, o2: any):
    """
    Check if two things are almost equal. Must be able to add and subtract them.
    """

    try:
        # Trick to cast them to the same type
        x = o1 + o2
        y = o1 - o2
        x = (x + y) / 2
        y = y - x
    except TypeError:
        raise

    if isinstance(x, float) or isinstance(x, np.floating):
        return np.abs(x - y) < _global_epsilon
    if isinstance(y, int) or isinstance(x, np.integer):
        return x == y
    elif type(x) == np.ndarray:
        return np.linalg.norm(x - y, _comparison_norm) < _global_epsilon
    else:
        raise TypeError(f"Unsupported type '{type(o1)}' for comparison")

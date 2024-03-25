import numpy as np
from zonopt import global_config as config
import math

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
        return np.abs(x - y) < config.comparison_epsilon
    if isinstance(y, int) or isinstance(x, np.integer):
        return x == y
    elif type(x) == np.ndarray:
        return np.linalg.norm(x - y, config.comparison_metric) < config.comparison_epsilon
    else:
        raise TypeError(f"Unsupported type '{type(o1)}' for comparison")

from zonopt.polytope.comparisons import almost_equal
from zonopt.config import global_config
import numpy as np
import unittest


class TestComparisons(unittest.TestCase):
    def setUp(self):
        global_config.comparison_epsilon = 0.01

    def test_int(self):
        a, b = 1, 2
        c, d = 3, 3

        self.assertFalse(almost_equal(a, b))
        self.assertTrue(almost_equal(c, d))

    def test_float(self):
        a, b = 1.3, 1.333
        c, d = 1.333, 1.334

        self.assertFalse(almost_equal(a, b))
        self.assertTrue(almost_equal(c, d))

    def test_array(self):
        a, b = np.array([1.00, -3.33]), np.array([1.5, 3.3])
        c, d = np.array([1.000, -3.333]), np.array([1.001, -3.334])

        self.assertFalse(almost_equal(a, b))
        self.assertTrue(almost_equal(c, d))

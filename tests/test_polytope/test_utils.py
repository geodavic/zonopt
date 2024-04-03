from zonopt.polytope.utils import subsets, is_centrally_symmetric
import numpy as np
import unittest


class TestSubsets(unittest.TestCase):
    def test_subsets(self):
        arr = [1, 2, 3]
        collection = subsets(arr)
        self.assertEqual(len(collection), 8)
        self.assertIn((1, 2, 3), collection)

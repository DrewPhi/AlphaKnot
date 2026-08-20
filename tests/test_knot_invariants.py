import unittest

from knot_invariants import jones_is_one, normalized_jones_in_a


class JonesPolynomialTests(unittest.TestCase):
    def test_one_crossing_unknot_normalizes_to_one(self):
        self.assertTrue(jones_is_one([[1, 1, 2, 2]]))

    def test_trefoil_polynomial_is_nontrivial(self):
        trefoil = [[1, 5, 2, 4], [3, 1, 4, 6], [5, 3, 6, 2]]
        self.assertEqual(
            normalized_jones_in_a(trefoil),
            {-16: -1, -12: 1, -4: 1},
        )
        self.assertFalse(jones_is_one(trefoil))

    def test_figure_eight_polynomial_is_nontrivial(self):
        figure_eight = [
            [8, 3, 1, 4],
            [2, 6, 3, 5],
            [6, 2, 7, 1],
            [4, 7, 5, 8],
        ]
        self.assertEqual(
            normalized_jones_in_a(figure_eight),
            {-8: 1, -4: -1, 0: 1, 4: -1, 8: 1},
        )
        self.assertFalse(jones_is_one(figure_eight))

import unittest
from itertools import product

import snappy

from knot_graph_game import KnotGraphGame
from pd_code_utils import crossing_sign, flip_crossing, pd_code_from_graph


class PDCodeTests(unittest.TestCase):
    def test_wraparound_crossing_is_positive(self):
        crossing = [7, 1, 8, 14]

        self.assertEqual(crossing_sign(crossing, 14), 1)
        self.assertEqual(flip_crossing(crossing, 14), [14, 7, 1, 8])

    def test_crossing_change_is_an_involution(self):
        crossings = (
            [1, 9, 2, 8],
            [7, 1, 8, 14],
            [14, 7, 1, 8],
        )
        for crossing in crossings:
            with self.subTest(crossing=crossing):
                self.assertEqual(
                    flip_crossing(flip_crossing(crossing, 14), 14),
                    crossing,
                )

    def test_invalid_upper_strand_orientation_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "consecutive oriented"):
            crossing_sign([1, 4, 2, 9], 14)

    def test_game_action_uses_positional_strand_pairing_for_wraparound(self):
        game = KnotGraphGame()
        board = game.getInitBoard()

        changed, _ = game.getNextState(board, 1, 2 * 3 + 1)

        self.assertEqual(pd_code_from_graph(changed)[3], [14, 7, 1, 8])

    def test_all_128_assignments_produce_valid_snapPy_pd_codes(self):
        game = KnotGraphGame()
        initial = game.getInitBoard()
        resolved_codes = set()

        for choices in product((0, 1), repeat=7):
            board = initial
            player = 1
            for node, choice in enumerate(choices):
                board, player = game.getNextState(
                    board, player, 2 * node + choice
                )
            pd_code = pd_code_from_graph(board)
            snappy.Link(pd_code)
            resolved_codes.add(tuple(tuple(crossing) for crossing in pd_code))

        self.assertEqual(len(resolved_codes), 128)

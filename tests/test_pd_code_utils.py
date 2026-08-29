import unittest
from itertools import product

import snappy

from knot_graph_game import KnotGraphGame
from pd_code_utils import (
    canonicalize_pd_code,
    crossing_sign,
    flip_crossing,
    pd_code_from_graph,
)


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

        canonical_action = game.source_action_to_canonical(2 * 3 + 1)
        canonical_crossing = canonical_action // 2
        changed, _ = game.getNextState(board, 1, canonical_action)

        expected = flip_crossing(game.pd_code[canonical_crossing], 14)
        self.assertEqual(pd_code_from_graph(changed)[canonical_crossing], expected)

    def test_action_attaches_resolution_state_to_crossing_node(self):
        game = KnotGraphGame()
        board = game.getInitBoard()

        kept, _ = game.getNextState(board, 1, 0)
        switched, _ = game.getNextState(board, 1, 1)

        self.assertEqual(kept.x[:, 4].tolist(), [1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(switched.x[:, 4].tolist(), [2, 0, 0, 0, 0, 0, 0])
        self.assertEqual(KnotGraphGame.reconstruct_pd_code(kept), game.pd_code)

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

    def test_canonical_action_mapping_round_trips(self):
        game = KnotGraphGame()
        game.getInitBoard()
        for action in range(game.getActionSize()):
            canonical = game.source_action_to_canonical(action)
            self.assertEqual(game.canonical_action_to_source(canonical), action)

    def test_canonicalization_is_idempotent(self):
        game = KnotGraphGame()
        game.getInitBoard()
        first = canonicalize_pd_code(game.source_pd_code)
        second = canonicalize_pd_code(first.pd_code)
        self.assertEqual(first.pd_code, second.pd_code)

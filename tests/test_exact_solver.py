import unittest

from exact_solver import ExactSolver, UNRESOLVED


class ExactSolverTests(unittest.TestCase):
    def setUp(self):
        self.solver = ExactSolver()

    def test_enumerates_all_partial_and_terminal_states(self):
        self.assertEqual(sum(1 for _ in self.solver.all_states()), 3 ** 7)
        self.assertEqual(sum(1 for _ in self.solver.all_states(terminal=True)), 2 ** 7)

    def test_optimal_actions_realize_exact_value(self):
        root = self.solver.initial_state
        expected = self.solver.value(root)
        self.assertTrue(self.solver.optimal_actions(root))
        for action in self.solver.optimal_actions(root):
            self.assertEqual(-self.solver.value(self.solver.play(root, action)), expected)

    def test_terminal_value_matches_winner_and_player_to_move(self):
        state = (0,) * 7
        expected = 1 if self.solver.winner(state) == -1 else -1
        self.assertEqual(self.solver.player_to_move(state), -1)
        self.assertEqual(self.solver.value(state), expected)

    def test_rejects_replaying_resolved_crossing(self):
        state = (0,) + (UNRESOLVED,) * 6
        with self.assertRaisesRegex(ValueError, "illegal action"):
            self.solver.play(state, 1)


if __name__ == "__main__":
    unittest.main()

import unittest

from capacity_test import exact_example
from exact_solver import ExactSolver, UNRESOLVED
from knot_graph_game import KnotGraphGame


class CapacityDatasetTests(unittest.TestCase):
    def setUp(self):
        self.game = KnotGraphGame()
        self.game.getInitBoard()
        self.solver = ExactSolver(self.game.initial_pd_code)

    def test_root_targets_all_optimal_actions_and_exact_value(self):
        graph = exact_example(
            self.game, self.solver, self.solver.initial_state
        )
        self.assertAlmostEqual(float(graph.target_policy.sum()), 1.0, places=6)
        self.assertTrue(graph.legal_mask.all())
        self.assertTrue((graph.target_policy > 0).all())
        self.assertEqual(float(graph.target_value.item()), -1.0)

    def test_resolved_crossing_actions_are_masked(self):
        state = (0,) + (UNRESOLVED,) * 6
        graph = exact_example(self.game, self.solver, state)
        self.assertFalse(bool(graph.legal_mask[0, 0]))
        self.assertFalse(bool(graph.legal_mask[0, 1]))
        self.assertEqual(float(graph.target_policy[0, :2].sum()), 0.0)
        self.assertAlmostEqual(float(graph.target_policy.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()

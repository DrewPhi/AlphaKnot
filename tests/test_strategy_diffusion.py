import unittest

import numpy as np
import torch
from torch_geometric.data import Batch

from knot_graph_game import KnotGraphGame
from knot_graph_nnet import PortGraphTransformerNet
from prime_knot_corpus import corpus_records
from strategy_diffusion_phate import diffusion_operator, matched_probe_states


class StrategyDiffusionTests(unittest.TestCase):
    def test_matched_panel_has_19_unique_nonterminal_states_at_every_size(self):
        for crossings in range(3, 9):
            states = matched_probe_states(crossings)
            self.assertEqual(len(states), 19)
            self.assertEqual(len(set(states)), 19)
            self.assertTrue(all(-1 in state for state in states))
            self.assertTrue(all(len(state) == crossings for state in states))

    def test_diffusion_operator_is_positive_and_row_stochastic(self):
        activations = np.arange(19 * 7, dtype=float).reshape(19, 7)
        operator = diffusion_operator(activations, knn=5)
        self.assertEqual(operator.shape, (19, 19))
        self.assertTrue(np.all(operator > 0))
        np.testing.assert_allclose(operator.sum(axis=1), np.ones(19))

    def test_value_penultimate_activation_has_shared_hidden_dimension(self):
        records = dict(corpus_records())
        small_game = KnotGraphGame(pd_code=records["3_1"])
        small = small_game.getInitBoard()
        large_game = KnotGraphGame(pd_code=records["8_1"])
        large = large_game.getInitBoard()
        model = PortGraphTransformerNet(
            small_game,
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            num_port_layers=1,
            dropout=0.0,
            position_mode="pd-structural",
            direct_state_residual=True,
            variable_size=True,
        )
        batch = Batch.from_data_list([small, large])
        policy, value, activations = model(batch, return_activations=True)
        self.assertEqual(tuple(policy.shape), (2, 16))
        self.assertEqual(tuple(value.shape), (2, 1))
        self.assertEqual(tuple(activations["value_penultimate"].shape), (2, 32))
        self.assertEqual(tuple(activations["graph_token"].shape), (2, 32))
        self.assertEqual(tuple(activations["crossing_tokens"].shape), (2, 8, 32))
        self.assertEqual(activations["crossing_mask"].dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()

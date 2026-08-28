import unittest

import torch
from torch_geometric.data import Batch

from exact_solver import ExactSolver
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import PDStateMLP, PortGraphTransformerNet
from seven_crossing_corpus import corpus_records, validate_corpus


class SevenCrossingCorpusTests(unittest.TestCase):
    def test_all_seven_snapPy_pd_codes_are_valid(self):
        validate_corpus()
        records = corpus_records()
        self.assertEqual([name for name, _ in records], [f"7_{i}" for i in range(1, 8)])

    def test_game_can_be_bound_to_one_corpus_pd_code(self):
        _, pd_code = corpus_records()[-1]
        game = KnotGraphGame(pd_code=pd_code)
        board = game.getInitBoard()
        self.assertEqual(game.initial_pd_code, pd_code)
        self.assertEqual(board.x[:, :4].long().tolist(), pd_code)

    def test_shared_model_accepts_different_pd_codes_in_one_batch(self):
        records = corpus_records()
        first_game = KnotGraphGame(pd_code=records[0][1])
        first = first_game.getInitBoard()
        second_game = KnotGraphGame(pd_code=records[-1][1])
        second = second_game.getInitBoard()
        model = PortGraphTransformerNet(
            first_game,
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            num_port_layers=1,
            dropout=0.0,
            position_mode="pd-traversal",
            direct_state_residual=True,
        )
        policy, value = model(Batch.from_data_list([first, second]))
        self.assertEqual(tuple(policy.shape), (2, 14))
        self.assertEqual(tuple(value.shape), (2, 1))

    def test_shadow_group_metadata_is_not_offset_during_batching(self):
        records = corpus_records()
        graphs = []
        for group, (_, pd_code) in enumerate(records[:2]):
            game = KnotGraphGame(pd_code=pd_code)
            graph = game.getInitBoard()
            graph.shadow_group = torch.tensor([group], dtype=torch.long)
            graphs.append(graph)
        batch = Batch.from_data_list(graphs)
        self.assertEqual(batch.shadow_group.tolist(), [0, 1])

    def test_pd_state_mlp_accepts_multiple_pd_codes(self):
        records = corpus_records()
        game = KnotGraphGame(pd_code=records[0][1])
        first = game.getInitBoard()
        second_game = KnotGraphGame(pd_code=records[1][1])
        second = second_game.getInitBoard()
        model = PDStateMLP(game, hidden_dim=64, num_layers=2)
        policy, value = model(Batch.from_data_list([first, second]))
        self.assertEqual(tuple(policy.shape), (2, 14))
        self.assertEqual(tuple(value.shape), (2, 1))

    def test_each_pd_code_has_the_expected_exact_state_count(self):
        for _, pd_code in corpus_records():
            solver = ExactSolver(pd_code)
            self.assertEqual(sum(1 for _ in solver.all_states(terminal=False)), 2059)


if __name__ == "__main__":
    unittest.main()

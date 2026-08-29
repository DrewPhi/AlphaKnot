import unittest

import torch
from torch_geometric.data import Batch

from exact_solver import ExactSolver
from evaluate_equivalent_pd import equivalent_variants
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import PDStateMLP, PortGraphTransformerNet
from pd_code_utils import canonicalize_pd_code, crossing_sign
from prime_knot_corpus import corpus_records as prime_corpus_records
from prime_knot_corpus import validate_corpus as validate_prime_corpus
from seven_crossing_corpus import corpus_records, validate_corpus
from variable_size_capacity_test import aggregate, dense_action_targets


class SevenCrossingCorpusTests(unittest.TestCase):
    def test_all_seven_snapPy_pd_codes_are_valid(self):
        validate_corpus()
        records = corpus_records()
        self.assertEqual([name for name, _ in records], [f"7_{i}" for i in range(1, 8)])

    def test_game_can_be_bound_to_one_corpus_pd_code(self):
        _, pd_code = corpus_records()[-1]
        game = KnotGraphGame(pd_code=pd_code)
        board = game.getInitBoard()
        canonical = canonicalize_pd_code(pd_code).as_lists()
        self.assertEqual(game.initial_pd_code, canonical)
        self.assertEqual(board.x[:, :4].long().tolist(), canonical)

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

    def test_equivalent_serializations_remain_valid_seven_crossing_codes(self):
        _, pd_code = corpus_records()[0]
        expected = canonicalize_pd_code(pd_code).pd_code
        expected_solver_code = ExactSolver(pd_code).pd_code
        expected_graph = KnotGraphGame(pd_code=pd_code).getInitBoard()
        for variant in equivalent_variants(pd_code).values():
            game = KnotGraphGame(pd_code=variant)
            board = game.getInitBoard()
            self.assertEqual(tuple(board.x.shape), (7, 5))
            self.assertEqual(
                sorted(int(label) for label in board.x[:, :4].flatten()),
                sorted(list(range(1, 15)) * 2),
            )
            for crossing in variant:
                crossing_sign(crossing, 14)
            self.assertEqual(canonicalize_pd_code(variant).pd_code, expected)
            self.assertEqual(ExactSolver(variant).pd_code, expected_solver_code)
            self.assertTrue(torch.equal(board.x, expected_graph.x))
            self.assertTrue(torch.equal(board.edge_index, expected_graph.edge_index))
            self.assertTrue(torch.equal(board.edge_attr, expected_graph.edge_attr))


class PrimeKnotCorpusTests(unittest.TestCase):
    def test_standard_prime_table_through_eight_crossings_is_complete_and_valid(self):
        validate_prime_corpus()
        records = prime_corpus_records()
        self.assertEqual(len(records), 35)
        self.assertEqual(records[0][0], "3_1")
        self.assertEqual(records[-1][0], "8_21")
        self.assertEqual(
            {
                crossings: sum(len(pd_code) == crossings for _, pd_code in records)
                for crossings in range(3, 9)
            },
            {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21},
        )

    def test_variable_model_batches_three_and_eight_crossing_games(self):
        records = dict(prime_corpus_records())
        small_game = KnotGraphGame(pd_code=records["3_1"])
        small = small_game.getInitBoard()
        large_game = KnotGraphGame(pd_code=records["8_21"])
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
        policy, value = model(Batch.from_data_list([small, large]))
        self.assertEqual(tuple(policy.shape), (2, 16))
        self.assertEqual(tuple(value.shape), (2, 1))

    def test_mixed_size_exact_targets_pad_to_batch_action_width(self):
        records = dict(prime_corpus_records())
        graphs = []
        for name in ("3_1", "8_1"):
            game = KnotGraphGame(pd_code=records[name])
            graph = game.getInitBoard()
            graph.node_target_policy = torch.zeros(len(records[name]), 2)
            graph.node_legal_mask = torch.ones(len(records[name]), 2, dtype=torch.bool)
            graphs.append(graph)
        batch = Batch.from_data_list(graphs)
        target, legal = dense_action_targets(batch)
        self.assertEqual(tuple(target.shape), (2, 16))
        self.assertEqual(tuple(legal.shape), (2, 16))
        self.assertFalse(legal[0, 6:].any())
        self.assertTrue(legal[1].all())

    def test_metric_aggregation_accepts_one_pass_iterables(self):
        rows = (
            {"states": 2, "policy_correct": 1, "value_correct": 2, "optimal_mass": 1.5},
            {"states": 3, "policy_correct": 2, "value_correct": 1, "optimal_mass": 2.0},
        )
        self.assertEqual(
            aggregate(iter(rows)),
            {"states": 5, "policy_correct": 3, "value_correct": 3, "optimal_mass": 3.5},
        )


if __name__ == "__main__":
    unittest.main()

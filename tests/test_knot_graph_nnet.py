import unittest

import torch

from knot_graph_game import KnotGraphGame
from knot_graph_nnet import (
    CrossingStateMLP,
    KnotGraphNet,
    PortGraphTransformerNet,
)


class OrderedPDCrossingEncodingTests(unittest.TestCase):
    def test_permuting_pd_slots_changes_crossing_encoding(self):
        game = KnotGraphGame()
        game.getInitBoard()
        model = KnotGraphNet(
            game,
            hidden_dim=8,
            num_heads=1,
            num_layers=1,
            dropout=0.0,
        )

        # Make the assertion deterministic: output coordinate zero reads the
        # first embedding coordinate from PD slot zero.
        with torch.no_grad():
            model.embed_pos.weight.zero_()
            model.embed_strand.weight.zero_()
            model.embed_strand.weight[:, 0] = torch.arange(
                model.embed_strand.num_embeddings,
                dtype=model.embed_strand.weight.dtype,
            )
            model.crossing_projection.weight.zero_()
            model.crossing_projection.bias.zero_()
            model.crossing_projection.weight[0, 0] = 1.0

        original = torch.tensor([[1, 9, 2, 8]])
        same_labels_different_order = torch.tensor([[9, 1, 2, 8]])

        original_encoding = model.encode_crossings(original)
        permuted_encoding = model.encode_crossings(same_labels_different_order)

        self.assertNotEqual(
            original_encoding[0, 0].item(),
            permuted_encoding[0, 0].item(),
        )

    def test_forward_uses_direct_crossing_resolution_state(self):
        game = KnotGraphGame()
        unresolved = game.getInitBoard()
        resolved, _ = game.getNextState(unresolved, 1, 0)
        model = KnotGraphNet(
            game,
            hidden_dim=8,
            num_heads=1,
            num_layers=1,
            dropout=0.0,
        )
        model.eval()

        with torch.no_grad():
            model.embed_crossing_state.weight.zero_()
            model.embed_crossing_state.weight[1].fill_(1.0)
            unresolved_policy, unresolved_value = model(unresolved)
            resolved_policy, resolved_value = model(resolved)

        self.assertFalse(torch.equal(unresolved_policy, resolved_policy))
        self.assertFalse(torch.equal(unresolved_value, resolved_value))


class CrossingStateMLPTests(unittest.TestCase):
    def test_batched_output_shapes(self):
        game = KnotGraphGame()
        first = game.getInitBoard()
        second, _ = game.getNextState(first, 1, 0)
        from torch_geometric.data import Batch

        batch = Batch.from_data_list([first, second])
        model = CrossingStateMLP(game, hidden_dim=32, state_embed_dim=4)
        policy, value = model(batch)

        self.assertEqual(tuple(policy.shape), (2, game.getActionSize()))
        self.assertEqual(tuple(value.shape), (2, 1))
        self.assertFalse(torch.equal(policy[0], policy[1]))


class PortGraphTransformerTests(unittest.TestCase):
    def _model(
        self,
        game,
        position_mode="none",
        direct_state_residual=False,
    ):
        model = PortGraphTransformerNet(
            game,
            hidden_dim=32,
            num_heads=4,
            num_layers=2,
            num_port_layers=2,
            dropout=0.0,
            position_mode=position_mode,
            direct_state_residual=direct_state_residual,
        )
        model.eval()
        return model

    def test_batched_output_shapes(self):
        from torch_geometric.data import Batch

        game = KnotGraphGame()
        first = game.getInitBoard()
        second, _ = game.getNextState(first, 1, 0)
        policy, value = self._model(game)(Batch.from_data_list([first, second]))

        self.assertEqual(tuple(policy.shape), (2, game.getActionSize()))
        self.assertEqual(tuple(value.shape), (2, 1))

    def test_arc_label_renumbering_does_not_change_predictions(self):
        game = KnotGraphGame()
        board = game.getInitBoard()
        relabeled = board.clone()
        permutation = torch.tensor(
            [0, 8, 3, 12, 5, 14, 1, 10, 7, 2, 13, 6, 11, 4, 9]
        )
        relabeled.x[:, :4] = permutation[relabeled.x[:, :4].long()]
        model = self._model(game)

        with torch.no_grad():
            policy, value = model(board)
            relabeled_policy, relabeled_value = model(relabeled)

        self.assertTrue(torch.allclose(policy, relabeled_policy, atol=1e-6))
        self.assertTrue(torch.allclose(value, relabeled_value, atol=1e-6))

    def test_pd_traversal_positions_follow_first_encounter_not_list_order(self):
        game = KnotGraphGame()
        board = game.getInitBoard()
        permutation = torch.tensor([2, 0, 6, 1, 5, 3, 4])
        permuted_labels = board.x[permutation, :4].long()
        model = self._model(game, position_mode="pd-traversal")

        original_positions = model._crossing_positions(board.x[:, :4].long())
        positions = model._crossing_positions(permuted_labels)

        self.assertEqual(
            positions.tolist(), original_positions[permutation].tolist()
        )
        self.assertNotEqual(positions.tolist(), list(range(7)))

    def test_pd_position_model_is_equivariant_to_crossing_list_permutation(self):
        game = KnotGraphGame()
        board = game.getInitBoard()
        permutation = torch.tensor([2, 0, 6, 1, 5, 3, 4])
        permuted = board.clone()
        permuted.x = board.x[permutation].clone()
        model = self._model(
            game,
            position_mode="pd-traversal",
            direct_state_residual=True,
        )

        with torch.no_grad():
            policy, value = model(board)
            permuted_policy, permuted_value = model(permuted)

        policy = policy.reshape(7, 2)
        permuted_policy = permuted_policy.reshape(7, 2)
        self.assertTrue(
            torch.allclose(permuted_policy, policy[permutation], atol=1e-6)
        )
        self.assertTrue(torch.allclose(value, permuted_value, atol=1e-6))

    def test_structural_positions_are_equivariant_to_crossing_list_permutation(self):
        game = KnotGraphGame()
        board = game.getInitBoard()
        batch = torch.zeros(board.x.size(0), dtype=torch.long)
        permutation = torch.tensor([2, 0, 6, 1, 5, 3, 4])
        features = PortGraphTransformerNet._structural_position_features(
            board.x[:, :4].long(), batch
        )
        permuted = PortGraphTransformerNet._structural_position_features(
            board.x[permutation, :4].long(), batch
        )
        self.assertTrue(torch.allclose(permuted, features[permutation]))


if __name__ == "__main__":
    unittest.main()

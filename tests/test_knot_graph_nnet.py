import unittest

import torch

from knot_graph_game import KnotGraphGame
from knot_graph_nnet import CrossingStateMLP, KnotGraphNet


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


if __name__ == "__main__":
    unittest.main()

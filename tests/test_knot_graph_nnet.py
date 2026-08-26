import unittest

import torch

from knot_graph_game import KnotGraphGame
from knot_graph_nnet import KnotGraphNet


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


if __name__ == "__main__":
    unittest.main()

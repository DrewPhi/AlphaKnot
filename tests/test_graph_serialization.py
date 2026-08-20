import unittest

import torch

from knot_graph_game import KnotGraphGame
from pd_code_utils import deserialize_graph, serialize_graph


class GraphSerializationTests(unittest.TestCase):
    def test_round_trip_preserves_graph_tensors(self):
        game = KnotGraphGame()
        board = game.getInitBoard()

        restored = deserialize_graph(serialize_graph(board))

        self.assertTrue(torch.equal(board.x, restored.x))
        self.assertTrue(torch.equal(board.edge_index, restored.edge_index))
        self.assertTrue(torch.equal(board.edge_attr, restored.edge_attr))

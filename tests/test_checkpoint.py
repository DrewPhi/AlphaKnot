import tempfile
import unittest
from pathlib import Path

import torch

from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_restores_loss_and_model_parameters(self):
        game = KnotGraphGame()
        board = game.getInitBoard()
        source = NNetWrapper(game, device="cpu")
        source.latest_loss = 0.125

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth.tar"
            source.save_checkpoint(str(path))

            restored = NNetWrapper(game, device="cpu")
            restored.load_checkpoint(str(path))

        self.assertEqual(restored.latest_loss, 0.125)
        source_policy, source_value = source.predict(board)
        restored_policy, restored_value = restored.predict(board)
        self.assertTrue(torch.allclose(
            torch.from_numpy(source_policy), torch.from_numpy(restored_policy)
        ))
        self.assertAlmostEqual(source_value, restored_value)

import numpy as np
import torch
import unittest

from mcts import MCTS


class Args:
    numMCTSSims = 1
    cpuct = 1.0


class DummyGame:
    def stringRepresentation(self, state):
        return str(state)

    def getGameEnded(self, board, player):
        return 0

    def getValidMoves(self, board, player):
        return torch.tensor([1, 0, 1, 0], dtype=torch.uint8)

    def getActionSize(self):
        return 4


class DummyNetwork:
    def predict(self, board):
        return np.array([0.4, 0.3, 0.2, 0.1]), 0.25


class MCTSTests(unittest.TestCase):
    def test_root_noise_never_revives_invalid_actions(self):
        mcts = MCTS(DummyGame(), DummyNetwork(), Args(), add_root_noise=True)
        mcts.search("root", 1, is_root=True)

        policy = mcts.Ps[str(("root", 1))]
        self.assertEqual(policy[1], 0)
        self.assertEqual(policy[3], 0)
        self.assertTrue(np.isclose(policy.sum(), 1.0))

    def test_one_simulation_uses_legal_root_prior(self):
        mcts = MCTS(DummyGame(), DummyNetwork(), Args(), add_root_noise=False)
        policy = mcts.getActionProb("root", 1, temp=1)

        self.assertEqual(policy[1], 0)
        self.assertEqual(policy[3], 0)
        self.assertTrue(np.isclose(sum(policy), 1.0))

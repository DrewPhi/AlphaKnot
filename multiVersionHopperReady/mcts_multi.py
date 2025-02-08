# mcts_multi.py

import math
import numpy as np
import config  # Import config.py
from joblib import Parallel, delayed

class MCTS:
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.Qsa = {}  # Q values for state-action pairs
        self.Nsa = {}  # Visit counts for state-action pairs
        self.Ns = {}   # Visit counts for states
        self.Ps = {}   # Policy returned by neural network
        self.c_puct = config.c_puct  # Load exploration constant from config.py
        self.num_parallel_mcts = config.num_parallel_mcts  # Number of parallel MCTS simulations

    def search(self, game):
        """ Runs a single MCTS search from the given game state. """
        canonicalBoard = game.get_canonical_form()
        s = self._hash_state(canonicalBoard)

        if s not in self.Ps:
            # Leaf node
            if game.is_terminal():
                return -game.get_winner()
            
            valids = game.get_valid_moves()
            policy, v = self.nnet.predict(canonicalBoard)
            policy = policy * valids  # Mask invalid moves
            
            sum_policy = np.sum(policy)
            if sum_policy > 0:
                policy /= sum_policy
            else:
                policy = valids / np.sum(valids) if np.sum(valids) > 0 else valids

            self.Ps[s] = policy
            self.Ns[s] = 0
            return -v

        if game.is_terminal():
            return -game.get_winner()

        valids = game.get_valid_moves()
        ucb_values = np.full(game.get_action_size(), -np.inf)

        for a in range(game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    ucb_values[a] = self.Qsa[(s, a)] + self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    ucb_values[a] = self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

        best_a = np.argmax(ucb_values)
        
        # Clone game efficiently using weak references
        next_game = game.clone()  # Do NOT use weakref.proxy
        next_game.make_move(best_a)
        
        v = self.search(next_game)

        if (s, best_a) in self.Qsa:
            self.Qsa[(s, best_a)] = (self.Nsa[(s, best_a)] * self.Qsa[(s, best_a)] + v) / (self.Nsa[(s, best_a)] + 1)
            self.Nsa[(s, best_a)] += 1
        else:
            self.Qsa[(s, best_a)] = v
            self.Nsa[(s, best_a)] = 1
        
        self.Ns[s] += 1
        return -v

    def parallel_search(self, game):
        """ Runs multiple MCTS simulations in parallel using joblib. """
        results = Parallel(n_jobs=self.num_parallel_mcts)(
            delayed(self.search)(game.clone()) for _ in range(config.num_mcts_sims)
        )
        return results

    def _hash_state(self, state):
        return state.tobytes()

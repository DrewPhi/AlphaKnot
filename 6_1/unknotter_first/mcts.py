# mcts.py

import math
import numpy as np
import config

class MCTS:
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.Qsa = {}  # Q values for (state, action)
        self.Nsa = {}  # Visit count for (state, action)
        self.Ns = {}   # Visit count for states
        self.Ps = {}   # Policy for states
        self.c_puct = 2.0  # Increased for stronger exploration

    def search(self, game, is_root=False):
        canonicalBoard = game.get_canonical_form()
        s = self._hash_state(canonicalBoard)

        if s not in self.Ps:
            if game.is_terminal():
                v = game.get_winner()
                return -v

            valids = game.get_valid_moves()
            nn_input = game.get_nn_input()
            pi_cross, pi_res, v = self.nnet.predict(nn_input)
            policy = pi_cross[:, None] * pi_res[None, :]
            policy = policy.flatten()
            policy *= valids
            sum_policy = np.sum(policy)

            if sum_policy > 0:
                policy /= sum_policy
            else:
                policy = valids / np.sum(valids) if np.sum(valids) > 0 else valids

            if is_root:
                alpha = config.dirichlet_alpha
                epsilon = config.dirichlet_epsilon
                noise = np.random.dirichlet([alpha] * len(policy))
                policy = (1 - epsilon) * policy + epsilon * noise

            self.Ps[s] = policy
            self.Ns[s] = 0
            return -v

        if game.is_terminal():
            v = game.get_winner()
            return -v

        valids = game.get_valid_moves()
        best_ucb = -float('inf')
        best_a = -1

        for a in range(len(valids)):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.c_puct * self.Ps[s][a] * \
                        math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
                if u > best_ucb:
                    best_ucb = u
                    best_a = a

        a = best_a
        next_game = game.clone()
        next_game.make_move(a)
        v = self.search(next_game, is_root=False)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v


    def _hash_state(self, state):
        return state.tobytes()

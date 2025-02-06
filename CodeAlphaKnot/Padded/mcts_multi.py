

import math
import numpy as np
import config  

class MCTS:
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.Qsa = {}  
        self.Nsa = {}  
        self.Ns = {}   
        self.Ps = {}   
        self.c_puct = 1.0  

    def search(self, game):
        canonicalBoard = game.get_canonical_form()
        s = self._hash_state(canonicalBoard)
        if s not in self.Ps:
            
            if game.is_terminal():
              
                v = game.get_winner()
                return -v
            valids = game.get_valid_moves()
            policy, v = self.nnet.predict(canonicalBoard)
            policy = policy * valids  # Mask invalid moves
            sum_policy = np.sum(policy)
            if sum_policy > 0:
                policy /= sum_policy
            else:
            
                valids_sum = np.sum(valids)
                if valids_sum > 0:
                    policy = valids / valids_sum
                else:
               
                    return -v
            self.Ps[s] = policy
            self.Ns[s] = 0
            return -v
        if game.is_terminal():
        
            v = game.get_winner()
            return -v
        valids = game.get_valid_moves()
        best_ucb = -float('inf')
        best_a = -1
        for a in range(game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
                if u > best_ucb:
                    best_ucb = u
                    best_a = a
        a = best_a
      
        next_game = game.clone()
        next_game.make_move(a)
        v = self.search(next_game)
     
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

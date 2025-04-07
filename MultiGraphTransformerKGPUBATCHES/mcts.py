import math
import numpy as np
import torch

class MCTS:
    def __init__(self, game, nnet, args, add_root_noise=False):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.add_root_noise = add_root_noise

        self.Qsa = {}  # stores Q values for (s,a)
        self.Nsa = {}  # stores visit counts for (s,a)
        self.Ns = {}   # stores visit counts for s
        self.Ps = {}   # stores initial policy (returned by neural net)

        self.Es = {}   # stores game.getGameEnded(s)
        self.Vs = {}   # stores game.getValidMoves(s)

    def getActionProb(self, canonicalBoard,current_player, temp=1):
        state = (canonicalBoard, current_player)

        states = []
        for _ in range(self.args.numMCTSSims):
            canonical = self.game.getCanonicalForm(state[0], state[1])
            states.append(canonical)

        # Batch predict (returns list of pi, v pairs)
        pi_v_batch = self.nnet.predict_batch(states)

        for canonicalBoard, current_player, (pi, v) in zip(states, [state[1]]*len(states), pi_v_batch):
            self._process_leaf(canonicalBoard, current_player, pi, v)


        s = self.game.stringRepresentation(state)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        total = float(sum(counts))
        return [x / total for x in counts]


    def search(self, canonicalBoard, player):
        s = self.game.stringRepresentation((canonicalBoard, player))

        # Terminal state
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, player)
        if self.Es[s] != 0:
            return -self.Es[s]

        # Leaf node
        if s not in self.Ps:
            pi, v = self.nnet.predict(canonicalBoard)
            return self._process_leaf(canonicalBoard, player, pi, v)

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_state, next_player = self.game.getNextState(canonicalBoard, player, a)
        next_canonical, next_canonical_player = self.game.getCanonicalForm(next_state, next_player)

        v = self.search(next_canonical, next_canonical_player)

        # Backpropagation
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v


    def _process_leaf(self, canonicalBoard, player, pi, v):
        s = self.game.stringRepresentation((canonicalBoard, player))
        valids = self.game.getValidMoves(canonicalBoard, player).cpu().numpy()
        pi = pi * valids
        sum_pi = np.sum(pi)

        if sum_pi > 0:
            pi /= sum_pi
        else:
            pi = pi + valids
            pi /= np.sum(pi)

        if self.add_root_noise:
            epsilon = 0.25
            alpha = 0.3
            noise = np.random.dirichlet([alpha] * len(pi))
            pi = (1 - epsilon) * pi + epsilon * noise

        self.Ps[s] = pi
        self.Vs[s] = valids
        self.Ns[s] = 0
        return -v

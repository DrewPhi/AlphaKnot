import math
import numpy as np
import torch

class MCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        self.Qsa = {}  # stores Q values for (s,a)
        self.Nsa = {}  # stores visit counts for (s,a)
        self.Ns = {}   # stores visit counts for s
        self.Ps = {}   # stores initial policy (returned by neural net)

        self.Es = {}   # stores game.getGameEnded(s)
        self.Vs = {}   # stores game.getValidMoves(s)

    def getActionProb(self, canonicalBoard,current_player, temp=1):
        state = (canonicalBoard, current_player)

        for _ in range(self.args.numMCTSSims):
            self.search(state[0], state[1])

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

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, player)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            # First time seeing this state
            pi, v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, player).cpu().numpy()
            pi = pi * valids  # mask invalid moves
            sum_pi = np.sum(pi)

            if sum_pi > 0:
                pi /= sum_pi
            else:
                pi = pi + valids
                pi /= np.sum(pi)

            self.Ps[s] = pi
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

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

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v


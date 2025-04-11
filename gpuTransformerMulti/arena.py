import numpy as np
import multiprocessing as mp
from multiprocessing import get_context

def run_single_game(game_class, nnet1_path, nnet2_path, args):
    from knot_graph_game import KnotGraphGame
    from knot_graph_nnet import NNetWrapper
    from mcts import MCTS
    import torch
    import numpy as np

    game = game_class()

    nnet1 = NNetWrapper(game, device="cuda" if torch.cuda.is_available() else "cpu")
    nnet1.load_checkpoint(nnet1_path)

    nnet2 = NNetWrapper(game, device="cuda" if torch.cuda.is_available() else "cpu")
    nnet2.load_checkpoint(nnet2_path)

    def player_factory(nnet):
        def player(board, player_id):
            canonicalBoard, current_player = game.getCanonicalForm(board, player_id)
            pi = MCTS(game, nnet, args).getActionProb(canonicalBoard, current_player, temp=0)
            return np.argmax(pi)
        return player

    player1 = player_factory(nnet1)
    player2 = player_factory(nnet2)

    board = game.getInitBoard()
    currentPlayer = 1
    while True:
        action = player1(board, currentPlayer) if currentPlayer == 1 else player2(board, currentPlayer)
        board, currentPlayer = game.getNextState(board, currentPlayer, action)
        result = game.getGameEnded(board, currentPlayer)
        if result != 0:
            return int(result * currentPlayer)


class Arena:
    def __init__(self, nnet1_path, nnet2_path, game_class, args):
        self.nnet1_path = nnet1_path
        self.nnet2_path = nnet2_path
        self.game_class = game_class
        self.args = args

    def playGames_parallel(self, num_games=20, num_workers=None):
        num_workers = num_workers or min(mp.cpu_count(), num_games)
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.starmap(run_single_game, [
                (self.game_class, self.nnet1_path, self.nnet2_path, self.args)
                for _ in range(num_games)
            ])

        oneWon = results.count(1)
        twoWon = results.count(-1)
        draws = results.count(0)

        return oneWon, twoWon, draws

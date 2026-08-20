import numpy as np
import multiprocessing as mp
from multiprocessing import get_context
import os
import config


_ARENA_STATE = None


def _init_arena_worker(game_class, nnet1_path, nnet2_path, args):
    global _ARENA_STATE
    from knot_graph_nnet import NNetWrapper
    import torch

    game = game_class()
    game.getInitBoard()

    device = getattr(config, "arena_device", "cpu")
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    nnet1 = NNetWrapper(game, device=device)
    nnet1.load_checkpoint(nnet1_path, load_optimizer=False)

    nnet2 = NNetWrapper(game, device=device)
    nnet2.load_checkpoint(nnet2_path, load_optimizer=False)
    _ARENA_STATE = (game, nnet1, nnet2, args)


def run_single_game(_game_number):
    from mcts import MCTS

    game, nnet1, nnet2, args = _ARENA_STATE

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
        if num_games < 1:
            raise ValueError("num_games must be positive")
        allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
        configured = getattr(config, "arena_workers", 0)
        if configured > 0:
            allocated = min(allocated, configured)
        num_workers = num_workers or min(allocated, num_games)
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_arena_worker,
            initargs=(
                self.game_class,
                self.nnet1_path,
                self.nnet2_path,
                self.args,
            ),
        ) as pool:
            results = pool.map(run_single_game, range(num_games))

        oneWon = results.count(1)
        twoWon = results.count(-1)
        draws = results.count(0)

        return oneWon, twoWon, draws

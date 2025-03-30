import config
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from coach import Coach
from arena import Arena
import numpy as np
from mcts import MCTS
class Args:
    numMCTSSims = config.numMCTSSims
    cpuct = config.cpuct

def currentNetPlayer(game, nnet, board, player):
    canonicalBoard, current_player = game.getCanonicalForm(board, player)
    pi = MCTS(game, nnet, config, add_root_noise=False).getActionProb(canonicalBoard, current_player, temp=0)
    return np.argmax(pi)


def main():
    game = KnotGraphGame()
    game.getInitBoard()
    game.resumeTraining = config.resume_training

    nnet = NNetWrapper(game)

    args = Args()
    coach = Coach(game, nnet, args)

    coach.learn()

    # Optional final Arena evaluation
    arena = Arena(
        lambda board, player: currentNetPlayer(game, nnet, board, player),
        lambda board, player: currentNetPlayer(game, nnet, board, player),
        game
    )


    oneWon, twoWon, draws = arena.playGames(num=config.arenaCompare, verbose=False)
    print(f"Arena results => New Net wins: {twoWon}/{config.arenaCompare}")

if __name__ == "__main__":
    main()

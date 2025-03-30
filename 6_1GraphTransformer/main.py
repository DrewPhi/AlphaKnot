import config
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from coach import Coach
from arena import Arena
import numpy as np

class Args:
    numMCTSSims = config.numMCTSSims
    cpuct = config.cpuct

def currentNetPlayer(game, nnet, board):
    from mcts import MCTS
    mctsPlayer = MCTS(game, nnet, Args())
    canonicalBoard, current_player = game.getCanonicalForm(board, 1)
    pi = mctsPlayer.getActionProb(canonicalBoard, current_player, temp=0)
    return np.argmax(pi)

def main():
    game = KnotGraphGame()
    game.getInitBoard()
    nnet = NNetWrapper(game)

    args = Args()
    coach = Coach(game, nnet, args)

    coach.learn()

    # Optional final Arena evaluation
    arena = Arena(
        lambda board: currentNetPlayer(game, nnet, board),
        lambda board: currentNetPlayer(game, nnet, board),
        game
    )

    oneWon, twoWon, draws = arena.playGames(num=config.arenaCompare, verbose=False)
    print(f"Arena results => New Net wins: {twoWon}/{config.arenaCompare}")

if __name__ == "__main__":
    main()

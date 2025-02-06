
import os
import numpy as np
import matplotlib.pyplot as plt
from game import KnotGame
from neural_network import NNetWrapper
import config  
def play_game(nnet1, nnet2, starting_player):
    pd_code = [
        [[1, 7, 2, 6, 0], [6, 1, 7, 2, 0]],
        [[3, 10, 4, 11, 0], [10, 4, 11, 3, 0]],
        [[5, 3, 6, 2, 0], [2, 5, 3, 6, 0]],
        [[7, 1, 8, 12, 0], [1, 8, 12, 7, 0]],
        [[9, 4, 10, 5, 0], [4, 10, 5, 9, 0]],
        [[11, 9, 12, 8, 0], [8, 11, 9, 12, 0]]
    ]
    game = KnotGame(pd_code, starting_player)
    current_player = starting_player
    nnet = {1: nnet1, -1: nnet2}
    while not game.game_over:
        canonicalBoard = game.get_canonical_form()
        pi, _ = nnet[game.get_current_player()].predict(canonicalBoard)
        valid_moves = game.get_valid_moves()
        pi = pi * valid_moves 
        sum_pi = np.sum(pi)
        if sum_pi > 0:
            pi /= sum_pi
        else:
            pi = valid_moves / np.sum(valid_moves)
        action = np.argmax(pi)
        game.make_move(action)
    return game.winner

def main():
    model_path = 'currentModel/current.pth.tar'
    n_games = config.n_games

    pd_code = [
        [[1, 7, 2, 6, 0], [6, 1, 7, 2, 0]],
        [[3, 10, 4, 11, 0], [10, 4, 11, 3, 0]],
        [[5, 3, 6, 2, 0], [2, 5, 3, 6, 0]],
        [[7, 1, 8, 12, 0], [1, 8, 12, 7, 0]],
        [[9, 4, 10, 5, 0], [4, 10, 5, 9, 0]],
        [[11, 9, 12, 8, 0], [8, 11, 9, 12, 0]]
    ]
    game = KnotGame(pd_code)
    nnet1 = NNetWrapper(game)
    nnet1.load_checkpoint(model_path)
    nnet2 = NNetWrapper(game)
    nnet2.load_checkpoint(model_path)

    results = []
    for i in range(n_games):
        starting_player = 1 if i % 2 == 0 else -1  
        winner = play_game(nnet1, nnet2, starting_player)
        results.append(winner)
        print(f"Game {i+1}/{n_games} completed. Winner: Player {winner}")

    wins_player1 = results.count(1)
    wins_player2 = results.count(-1)

    labels = ['Player 1 Wins', 'Player -1 Wins']
    counts = [wins_player1, wins_player2]

    plt.bar(labels, counts, color=['blue', 'red'])
    plt.title(f'AI vs AI over {n_games} games')
    plt.xlabel('Result')
    plt.ylabel('Number of Wins')
    plt.show()

if __name__ == "__main__":
    main()

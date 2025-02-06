import os
import numpy as np
import matplotlib.pyplot as plt
from game_multi import KnotGame
from neural_network_multi import NNetWrapper
import config
import torch

def pad_pd_code(pd_code, max_crossings):
    dummy_crossing = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    padding_length = max_crossings - len(pd_code)
    pd_code.extend([dummy_crossing] * padding_length)
    return pd_code

def play_game(nnet, pd_code, starting_player):
    pd_code = pad_pd_code(pd_code.copy(), config.max_crossings)
    game = KnotGame([pd_code], starting_player)
    while not game.game_over:
        canonicalBoard = game.get_canonical_form()
        pi, _ = nnet.predict(canonicalBoard)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'currentModel/current.pth.tar'
    n_games = config.n_games
    pd_codes_to_test = [
        [[[1, 5, 2, 4,0], [4, 1, 5, 2,0]], [[3, 1, 4, 6,0], [1, 4, 6, 3,0]], [[5, 3, 6, 2,0], [2, 5, 3, 6,0]]],
        [[[4, 2, 5, 1,0], [1, 4, 2, 5,0]], [[8, 6, 1, 5,0], [5, 8, 6, 1,0]], [[6, 3, 7, 4,0], [3, 7, 4, 6,0]], [[2, 7, 3, 8,0], [7, 3, 8, 2,0]]],
        [[[2, 8, 3, 7,0], [7, 2, 8, 3,0]], [[4, 10, 5, 9,0], [9, 4, 10, 5,0]], [[6, 2, 7, 1,0], [1, 6, 2, 7,0]], [[8, 4, 9, 3,0], [3, 8, 4, 9,0]], [[10, 6, 1, 5,0], [5, 10, 6, 1,0]]],
        [[[1, 7, 2, 6, 0], [6, 1, 7, 2, 0]], [[3, 10, 4, 11, 0], [10, 4, 11, 3, 0]], [[5, 3, 6, 2, 0], [2, 5, 3, 6, 0]], [[7, 1, 8, 12, 0], [1, 8, 12, 7, 0]], [[9, 4, 10, 5, 0], [4, 10, 5, 9, 0]], [[11, 9, 12, 8, 0], [8, 11, 9, 12, 0]]],
        [[[2, 10, 3, 9, 0], [9, 2, 10, 3, 0]], [[4, 14, 5, 13, 0], [13, 4, 14, 5, 0]], [[6, 12, 7, 11, 0], [11, 6, 12, 7, 0]], [[8, 2, 9, 1, 0], [1, 8, 2, 9, 0]], [[10, 8, 11, 7, 0], [7, 10, 8, 11, 0]], [[12, 6, 13, 5, 0], [5, 12, 6, 13, 0]], [[14, 4, 1, 3, 0], [3, 14, 4, 1, 0]]],
        [[[1, 9, 2, 8, 0], [8, 1, 9, 2, 0]], [[3, 7, 4, 6, 0], [6, 3, 7, 4, 0]], [[5, 12, 6, 13, 0], [12, 6, 13, 5, 0]], [[7, 3, 8, 2, 0], [2, 7, 3, 8, 0]], [[9, 1, 10, 16, 0], [1, 10, 16, 9, 0]], [[11, 15, 12, 14, 0], [14, 11, 15, 12, 0]], [[13, 4, 14, 5, 0], [4, 14, 5, 13, 0]], [[15, 11, 16, 10, 0], [10, 15, 11, 16, 0]]]
    ]

    config.max_crossings = 8

    if not os.path.isfile(model_path):
        print(f"Model file not found at {model_path}. Please train the model first.")
        return

    sample_pd_code = pad_pd_code(pd_codes_to_test[0].copy(), config.max_crossings)
    sample_game = KnotGame([sample_pd_code])
    nnet = NNetWrapper(sample_game)
    nnet.load_checkpoint(model_path)

    starting_player = 1

    for pd_code in pd_codes_to_test:
        num_crossings = len(pd_code)
        if num_crossings > config.max_crossings:
            print(f"The PD code has {num_crossings} crossings, which exceeds the maximum allowed by config.max_crossings.")
            continue

        print(f"Testing on PD code with {num_crossings} crossings.")
        results = []
        for i in range(n_games):
            winner = play_game(nnet, pd_code, starting_player)
            results.append(winner)
            print(f"Game {i+1}/{n_games} completed. Winner: {'Player 1' if winner == 1 else 'Player -1'}")

        starting_player_wins = results.count(starting_player)
        second_player_wins = results.count(-starting_player)

        labels = [f'{starting_player}', f'{-starting_player}']
        counts = [starting_player_wins, second_player_wins]
        colors = ['blue', 'orange']

        plt.bar(labels, counts, color=colors)
        plt.title(f'Wins by Starting Player ({starting_player}) vs Second Player ({-starting_player})\nOver {n_games} games (PD code with {num_crossings} crossings)')
        plt.xlabel('Player')
        plt.ylabel('Number of Wins')
        plt.show()

if __name__ == "__main__":
    main()

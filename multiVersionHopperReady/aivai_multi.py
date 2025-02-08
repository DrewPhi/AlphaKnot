# aivai_multi.py

import os
import numpy as np
import matplotlib.pyplot as plt
from game_multi import KnotGame
from neural_network_multi import NNetWrapper
import config  # Import config.py
import torch

def play_game(nnet, pd_code, starting_player):
    # Initialize game with the provided PD code and starting player
    game = KnotGame([pd_code], starting_player)
    while not game.game_over:
        canonicalBoard = game.get_canonical_form()
        pi, _ = nnet.predict(canonicalBoard)
        valid_moves = game.get_valid_moves()
        pi = pi * valid_moves  # Mask invalid moves
        sum_pi = np.sum(pi)
        if sum_pi > 0:
            pi /= sum_pi
        else:
            # If all valid moves were masked, select a random valid move
            pi = valid_moves / np.sum(valid_moves)
        action = np.argmax(pi)
        game.make_move(action)
    return game.winner

def main():
    # Set the device for PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path to the trained model
    model_path = 'currentModel/current.pth.tar'
    n_games = config.n_games

    # Specify the PD codes and their names you want to test on
    pd_codes_to_test = [
        ('6_1', [
            [[1, 7, 2, 6, 0], [6, 1, 7, 2, 0]],
            [[3, 10, 4, 11, 0], [10, 4, 11, 3, 0]],
            [[5, 3, 6, 2, 0], [2, 5, 3, 6, 0]],
            [[7, 1, 8, 12, 0], [1, 8, 12, 7, 0]],
            [[9, 4, 10, 5, 0], [4, 10, 5, 9, 0]],
            [[11, 9, 12, 8, 0], [8, 11, 9, 12, 0]]
        ]),
        ('6_2', [
            [[1, 8, 2, 9, 0], [8, 2, 9, 1, 0]],
            [[3, 11, 4, 10, 0], [10, 3, 11, 4, 0]],
            [[5, 1, 6, 12, 0], [1, 6, 12, 5, 0]],
            [[7, 2, 8, 3, 0], [2, 8, 3, 7, 0]],
            [[9, 7, 10, 6, 0], [6, 9, 7, 10, 0]],
            [[11, 5, 12, 4, 0], [4, 11, 5, 12, 0]]
        ]),
        ('6_3', [
            [[4, 2, 5, 1, 0], [1, 4, 2, 5, 0]],
            [[8, 4, 9, 3, 0], [3, 8, 4, 9, 0]],
            [[12, 9, 1, 10, 0], [9, 1, 10, 12, 0]],
            [[10, 5, 11, 6, 0], [5, 11, 6, 10, 0]],
            [[6, 11, 7, 12, 0], [11, 7, 12, 6, 0]],
            [[2, 8, 3, 7, 0], [7, 2, 8, 3, 0]]
        ])
    ]

    # Ensure all PD codes have the same number of crossings
    num_crossings = len(pd_codes_to_test[0][1])
    for name, pd_code in pd_codes_to_test:
        if len(pd_code) != num_crossings:
            print("All PD codes must have the same number of crossings.")
            return

    # Load the model
    if os.path.isfile(model_path):
        # Initialize a game instance for network initialization
        game = KnotGame([pd_codes_to_test[0][1]])
        nnet = NNetWrapper(game)
        nnet.load_checkpoint(model_path)
    else:
        print(f"Model file not found at {model_path}. Please train the model first.")
        return

    # Set the starting player (1 for unknotter)
    starting_player = 1  # Unknotter goes first every time

    for name, pd_code in pd_codes_to_test:
        print(f"Testing on knot {name} with {num_crossings} crossings.")
        results = []
        for i in range(n_games):
            winner = play_game(nnet, pd_code, starting_player)
            results.append(winner)
            print(f"Game {i+1}/{n_games} completed. Winner: {'Unknotter' if winner == 1 else 'Knotter'}")
        # Calculate results
        unknotter_wins = results.count(1)
        knotter_wins = results.count(-1)

        # Plot the results
        labels = ['Unknotter Wins', 'Knotter Wins']
        counts = [unknotter_wins, knotter_wins]

        plt.bar(labels, counts, color=['green', 'purple'])
        plt.title(f'AI vs AI over {n_games} games on knot {name}')
        plt.xlabel('Result')
        plt.ylabel('Number of Wins')
        plt.show()

if __name__ == "__main__":
    main()

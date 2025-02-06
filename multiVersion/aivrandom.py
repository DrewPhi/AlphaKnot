# aivrandom_multi.py

import os
import numpy as np
import matplotlib.pyplot as plt
from game_multi import KnotGame
from neural_network_multi import NNetWrapper
import config  # Import config.py
import random

def play_game(nnet, starting_player):
    # Initialize game with the starting player and PD code for 6_1 knot
    pd_code = [
        [[1, 7, 2, 6, 0], [6, 1, 7, 2, 0]],
        [[3, 10, 4, 11, 0], [10, 4, 11, 3, 0]],
        [[5, 3, 6, 2, 0], [2, 5, 3, 6, 0]],
        [[7, 1, 8, 12, 0], [1, 8, 12, 7, 0]],
        [[9, 4, 10, 5, 0], [4, 10, 5, 9, 0]],
        [[11, 9, 12, 8, 0], [8, 11, 9, 12, 0]]
    ]
    game = KnotGame([pd_code], starting_player)
    moves_sequence = []  # To record the sequence of moves
    while not game.game_over:
        if game.get_current_player() == -1:
            # AI's turn (Player -1)
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
        else:
            # Random player's turn (Player 1)
            valid_moves = game.get_valid_moves()
            valid_actions = np.where(valid_moves == 1)[0]
            action = random.choice(valid_actions)
        # Record the move
        crossing_index = action // 2
        choice_index = action % 2
        moves_sequence.append((crossing_index, choice_index))
        game.make_move(action)
    return game.winner, moves_sequence

def main():
    model_path = 'currentModel/current.pth.tar'
    n_games = 100  # Set to 100 games

    # Load the model
    pd_code = [
        [[1, 7, 2, 6, 0], [6, 1, 7, 2, 0]],
        [[3, 10, 4, 11, 0], [10, 4, 11, 3, 0]],
        [[5, 3, 6, 2, 0], [2, 5, 3, 6, 0]],
        [[7, 1, 8, 12, 0], [1, 8, 12, 7, 0]],
        [[9, 4, 10, 5, 0], [4, 10, 5, 9, 0]],
        [[11, 9, 12, 8, 0], [8, 11, 9, 12, 0]]
    ]
    game = KnotGame([pd_code])
    nnet = NNetWrapper(game)
    nnet.load_checkpoint(model_path)

    # Play n_games
    results = []
    random_wins_sequences = []  # To store sequences where random player wins
    for i in range(n_games):
        starting_player = 1  # Random player (Player 1) goes first every time
        winner, moves_sequence = play_game(nnet, starting_player)
        results.append(winner)
        if winner == 1:
            # Random player won, save the sequence
            random_wins_sequences.append(moves_sequence)
        print(f"Game {i+1}/{n_games} completed. Winner: {'AI (Knotter)' if winner == -1 else 'Random Player (Unknotter)'}")

    # AI is Player -1
    wins_ai = results.count(-1)
    wins_random = results.count(1)

    # Plot the results
    labels = ['AI (Knotter) Wins', 'Random (Unknotter) Wins']
    counts = [wins_ai, wins_random]

    plt.bar(labels, counts, color=['blue', 'green'])
    plt.title(f'AI vs Random over {n_games} games')
    plt.xlabel('Result')
    plt.ylabel('Number of Wins')
    plt.show()

    # Print the sequences where the random player won
    print("\nSequences of moves where the random player won:")
    if random_wins_sequences:
        for idx, seq in enumerate(random_wins_sequences, 1):
            seq_str = ' '.join(f"({move[0]},{move[1]})" for move in seq)
            print(f"Game {idx}: {seq_str}")
    else:
        print("No games were won by the random player.")

if __name__ == "__main__":
    main()

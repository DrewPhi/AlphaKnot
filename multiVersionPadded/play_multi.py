# play_multi.py

from game_multi import KnotGame
from neural_network_multi import NNetWrapper
import numpy as np
import os
import config  # Import config.py

# Your provided general PD codes for multiple knots
pd_codes = [
    # '6_1' knot
    [[[1, 7, 2, 6, 0], [6, 1, 7, 2, 0]],
     [[3, 10, 4, 11, 0], [10, 4, 11, 3, 0]],
     [[5, 3, 6, 2, 0], [2, 5, 3, 6, 0]],
     [[7, 1, 8, 12, 0], [1, 8, 12, 7, 0]],
     [[9, 4, 10, 5, 0], [4, 10, 5, 9, 0]],
     [[11, 9, 12, 8, 0], [8, 11, 9, 12, 0]]],
    # '6_2' knot
    [[[1, 8, 2, 9, 0], [8, 2, 9, 1, 0]],
     [[3, 11, 4, 10, 0], [10, 3, 11, 4, 0]],
     [[5, 1, 6, 12, 0], [1, 6, 12, 5, 0]],
     [[7, 2, 8, 3, 0], [2, 8, 3, 7, 0]],
     [[9, 7, 10, 6, 0], [6, 9, 7, 10, 0]],
     [[11, 5, 12, 4, 0], [4, 11, 5, 12, 0]]],
    # '6_3' knot
    [[[4, 2, 5, 1, 0], [1, 4, 2, 5, 0]],
     [[8, 4, 9, 3, 0], [3, 8, 4, 9, 0]],
     [[12, 9, 1, 10, 0], [9, 1, 10, 12, 0]],
     [[10, 5, 11, 6, 0], [5, 11, 6, 10, 0]],
     [[6, 11, 7, 12, 0], [11, 7, 12, 6, 0]],
     [[2, 8, 3, 7, 0], [7, 2, 8, 3, 0]]]
]

if __name__ == "__main__":
    starting_player = -1 if config.knotter_first else 1
    # For playing, let's select a PD code (or you can choose randomly)
    pd_code = pd_codes[0]  # Select the first PD code
    game = KnotGame([pd_code], starting_player)
    nnet = NNetWrapper(game)
    # Load the model
    model_path = 'currentModel/current.pth.tar'
    nnet.load_checkpoint(model_path)
    mcts = None  # We'll use the neural network directly for simplicity

    # Decide if the human is the knotter or unknotter
    # Let's assume the human is always Player 1
    human_player = 1
    ai_player = -1

    # Inform the player of their goal
    if human_player == starting_player:
        # Human goes first
        if human_player == 1:
            print("You are Player 1 and will go first.")
            print("Your goal is to resolve the crossings so that the resulting knot is an **unknot**.")
        else:
            print("You are Player -1 and will go first.")
            print("Your goal is to resolve the crossings so that the resulting knot is a **knot** (not the unknot).")
    else:
        # AI goes first
        if human_player == 1:
            print("You are Player 1 and will go second.")
            print("Your goal is to resolve the crossings so that the resulting knot is an **unknot**.")
        else:
            print("You are Player -1 and will go second.")
            print("Your goal is to resolve the crossings so that the resulting knot is a **knot** (not the unknot).")
    print("At each turn, enter your move as (crossing_index, choice_index).")
    print("For example, (0,1) selects the second choice for the first crossing.")
    print()

    while not game.game_over:
        print("\nCurrent game state:")
        # Print the current generalized PD code
        for i, crossing in enumerate(game.current_pd_code):
            selected_option = None
            if crossing[0][4] == 1:
                selected_option = 0
            elif crossing[1][4] == 1:
                selected_option = 1
            else:
                selected_option = None
            print(f"Crossing {i}:")
            print(f"  Option 0: {crossing[0][:4]}{' (selected)' if selected_option == 0 else ''}")
            print(f"  Option 1: {crossing[1][:4]}{' (selected)' if selected_option == 1 else ''}")
        if game.get_current_player() == human_player:
            # Human's turn
            valid = False
            while not valid:
                user_input = input("Enter your move as (crossing_index, choice_index): ")
                try:
                    crossing_index_str, choice_index_str = user_input.strip("() ").split(",")
                    crossing_index = int(crossing_index_str)
                    choice_index = int(choice_index_str)
                    # Check if the move is valid
                    if crossing_index < 0 or crossing_index >= len(game.current_pd_code):
                        print("Invalid crossing index. Please try again.")
                        continue
                    if choice_index not in [0, 1]:
                        print("Choice index must be 0 or 1. Please try again.")
                        continue
                    # Check if the crossing is unresolved
                    crossing = game.current_pd_code[crossing_index]
                    if crossing[0][4] == 0 and crossing[1][4] == 0:
                        action = crossing_index * 2 + choice_index
                        valid_moves = game.get_valid_moves()
                        if valid_moves[action]:
                            valid = True
                        else:
                            print("Invalid move. Please try again.")
                    else:
                        print("This crossing has already been resolved. Choose another one.")
                except Exception as e:
                    print("Invalid input format. Please enter in the format (crossing_index, choice_index).")
            game.make_move(action)
        else:
            # AI's turn
            print("AI is thinking...")
            canonicalBoard = game.get_canonical_form()
            pi, v = nnet.predict(canonicalBoard)
            valid_moves = game.get_valid_moves()
            pi = pi * valid_moves  # Mask invalid moves
            sum_pi = np.sum(pi)
            if sum_pi > 0:
                pi /= sum_pi
            else:
                # If all valid moves were masked, select a random valid move
                pi = valid_moves / np.sum(valid_moves)
            action = np.argmax(pi)
            crossing_index = action // 2
            choice_index = action % 2
            print(f"AI selects crossing {crossing_index}, choice {choice_index}.")
            game.make_move(action)

    # Game over
    print("\nGame over.")
    # Determine if the final code is a knot or an unknot
    final_pd_code = []
    for crossing in game.current_pd_code:
        if crossing[0][4] == 1:
            final_pd_code.append(crossing[0][:4])
        elif crossing[1][4] == 1:
            final_pd_code.append(crossing[1][:4])
    is_unknot = game.is_unknot(final_pd_code)
    if is_unknot:
        print("The final code is an unknot.")
    else:
        print("The final code is a knot.")

    if game.winner == human_player:
        print("You win!")
    else:
        print("AI wins.")

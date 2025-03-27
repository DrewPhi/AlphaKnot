# play.py

from game import KnotGame
from neural_network import NNetWrapper
import numpy as np
import os
import config  # Import config.py
import snappy
import spherogram

# Load the trained model
model_path = 'currentModel/current.pth.tar'
pd_code = [
    [[1, 7, 2, 6, 0], [6, 1, 7, 2, 0]],
    [[3, 10, 4, 11, 0], [10, 4, 11, 3, 0]],
    [[5, 3, 6, 2, 0], [2, 5, 3, 6, 0]],
    [[7, 1, 8, 12, 0], [1, 8, 12, 7, 0]],
    [[9, 4, 10, 5, 0], [4, 10, 5, 9, 0]],
    [[11, 9, 12, 8, 0], [8, 11, 9, 12, 0]]
]

def display_current_knot(game):
    # Generate two PD codes:
    # - One with unresolved crossings set to Option 0
    # - One with unresolved crossings set to Option 1
    pd_code_option_0 = []
    pd_code_option_1 = []

    for crossing in game.current_pd_code:
        if crossing[0][4] == 1:
            # Crossing resolved to Option 0
            pd_code_option_0.append(crossing[0][:4])
            pd_code_option_1.append(crossing[0][:4])
        elif crossing[1][4] == 1:
            # Crossing resolved to Option 1
            pd_code_option_0.append(crossing[1][:4])
            pd_code_option_1.append(crossing[1][:4])
        else:
            # Crossing unresolved, set to Option 0 and Option 1 respectively
            pd_code_option_0.append(crossing[0][:4])
            pd_code_option_1.append(crossing[1][:4])

    # Create knot diagrams
    try:
        link_option_0 = spherogram.Link(pd_code_option_0)
        link_option_1 = spherogram.Link(pd_code_option_1)
    except Exception as e:
        print(f"Error creating knot diagrams: {e}")
        return

    # Display the diagrams using link.view()
    print("Displaying knots with unresolved crossings set to Option 0 and Option 1.")
    print("Close the knot diagram window to continue the game.")
    print()

    print("Knot with unresolved crossings set to Option 0:")
    link_option_0.view()

    print("Knot with unresolved crossings set to Option 1:")
    link_option_1.view()

if __name__ == "__main__":
    starting_player = -1 if config.knotter_first else 1
    game = KnotGame(pd_code, starting_player)
    nnet = NNetWrapper(game)
    # Load the model
    nnet.load_checkpoint(model_path)
    mcts = None  # We'll use the neural network directly for simplicity

    # Decide if the human is the knotter or unknotter
    # Let's set the human to be Player -1 (knotter)
    human_player = 1
    ai_player = -1

    # Inform the player of their goal
    if human_player == starting_player:
        # Human goes first
        print("You are Player -1 (knotter) and will go first.")
        print("Your goal is to resolve the crossings so that the resulting knot is a **non-trivial knot** (not the unknot).")
    else:
        # AI goes first
        print("You are Player -1 (knotter) and will go second.")
        print("Your goal is to resolve the crossings so that the resulting knot is a **non-trivial knot** (not the unknot).")
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
        # Display the current knot diagrams
        display_current_knot(game)
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

    if (game.winner == human_player):
        print("You win!")
    else:
        print("AI wins.")

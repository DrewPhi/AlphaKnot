#!/usr/bin/env python3
import torch
import numpy as np
import os

from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from mcts import MCTS
import config

# Use a deterministic move selector for the model (player -1)
def model_move(game, model, board, player):
    canonicalBoard, current_player = game.getCanonicalForm(board, player)
    # Use MCTS with no added noise and temp=0 (deterministic)
    mcts = MCTS(game, model, config, add_root_noise=False)
    pi = mcts.getActionProb(canonicalBoard, current_player, temp=0)
    return int(np.argmax(pi))

# Recursive DFS that explores every legal move for the opponent (player 1)
def dfs(state, currentPlayer, move_sequence, game, model):
    # Check if the game has ended.
    # (Note: following the Arena convention, we multiply the game outcome by currentPlayer so that +1 means player1 win.)
    outcome = game.getGameEnded(state, currentPlayer)
    if outcome != 0:
        final_outcome = outcome * currentPlayer
        return final_outcome, move_sequence

    valid_moves_tensor = game.getValidMoves(state, currentPlayer)
    valid_moves = valid_moves_tensor.cpu().numpy()  # array of 0/1 for each action
    if valid_moves.sum() == 0:
        # Should not happen but return a draw if no moves available
        return 0, move_sequence

    # If it is the opponent's turn, enumerate all valid moves.
    if currentPlayer == 1:
        best_outcome = -float("inf")
        best_sequence = None
        for action in range(game.getActionSize()):
            if valid_moves[action]:
                new_state, nextPlayer = game.getNextState(state, currentPlayer, action)
                outcome_branch, sequence_branch = dfs(new_state, nextPlayer, move_sequence + [(currentPlayer, action)], game, model)
                # We are testing if the opponent can force a win (outcome > 0 means opponent wins).
                if outcome_branch > best_outcome:
                    best_outcome = outcome_branch
                    best_sequence = sequence_branch
                # Early exit if a winning branch is found.
                if best_outcome > 0:
                    return best_outcome, best_sequence
        return best_outcome, best_sequence
    else:
        # Model's turn: choose its move deterministically.
        action = model_move(game, model, state, currentPlayer)
        new_state, nextPlayer = game.getNextState(state, currentPlayer, action)
        return dfs(new_state, nextPlayer, move_sequence + [(currentPlayer, action)], game, model)

def main():
    # Initialize the game and get the starting board.
    game = KnotGraphGame()
    initial_state = game.getInitBoard()

    # Instantiate the neural net wrapper.
    model = NNetWrapper(game)
    
    # Optionally, try loading a pre-trained model checkpoint.
    checkpoint_path = os.path.join(config.checkpoint, 'best.pth.tar')
    if os.path.isfile(checkpoint_path):
        try:
            model.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print("Error loading checkpoint; using untrained model.", e)
    else:
        print("No checkpoint found; using current model weights (likely untrained).")
    
    # Since we want the model to play second, the opponent is player 1.
    starting_player = 1
    outcome, sequence = dfs(initial_state, starting_player, [], game, model)
    
    # If outcome > 0 then player1 wins.
    if outcome >= 0:
        print("Found a sequence where the opponent (player 1) wins!")
        print("Move sequence (player, action index):")
        for move in sequence:
            print(move)
    else:
        print("The model (player -1) is unbeatable against all opponent moves.")

if __name__ == "__main__":
    main()

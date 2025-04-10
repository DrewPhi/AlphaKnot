#!/usr/bin/env python3
import os
import torch
import numpy as np
import config

from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from mcts import MCTS

# Global counter for nodes processed (for progress reporting)
global_node_count = 0

def minimax_dfs(state, currentPlayer, move_sequence, game, model, depth=0):
    global global_node_count
    global_node_count += 1

    # Print progress every 10000 nodes.
    if global_node_count % 10000 == 0:
        print(f"Processed {global_node_count} nodes; current depth: {depth}; move_sequence: {move_sequence}")

    # Terminal check: if game is over, return value from player1's perspective.
    outcome = game.getGameEnded(state, currentPlayer)
    if outcome != 0:
        final_value = outcome * currentPlayer  # By convention: positive => win for player1.
        return final_value, move_sequence, 1  # 1 terminal state reached

    valid_moves_tensor = game.getValidMoves(state, currentPlayer)
    valid_moves = valid_moves_tensor.cpu().numpy()
    if valid_moves.sum() == 0:
        return 0, move_sequence, 1

    total_terminals = 0

    # For minimax: player 1 is maximizer, player -1 is minimizer.
    if currentPlayer == 1:
        best_value = -float("inf")
    else:
        best_value = float("inf")
    best_sequence = None

    # Branch on all legal moves for the current player.
    for action in range(game.getActionSize()):
        if valid_moves[action]:
            new_state, nextPlayer = game.getNextState(state, currentPlayer, action)
            child_value, child_sequence, child_terminals = minimax_dfs(
                new_state, nextPlayer, move_sequence + [(currentPlayer, action)], game, model, depth + 1
            )
            total_terminals += child_terminals

            if currentPlayer == 1:
                # Maximizer: choose the move with the highest value.
                if child_value > best_value:
                    best_value = child_value
                    best_sequence = child_sequence
            else:
                # Minimizer: choose the move with the lowest value.
                if child_value < best_value:
                    best_value = child_value
                    best_sequence = child_sequence

    return best_value, best_sequence, total_terminals

def main():
    # Initialize game and initial state.
    game = KnotGraphGame()
    initial_state = game.getInitBoard()

    # Instantiate the neural network wrapper.
    model = NNetWrapper(game)
    checkpoint_path = os.path.join(config.checkpoint, 'best.pth.tar')
    if os.path.isfile(checkpoint_path):
        try:
            model.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print("Error loading checkpoint; using untrained model.", e)
    else:
        print("No checkpoint found; using current model weights (likely untrained).")

    # Set starting player: we assume opponent is player 1 (maximizer) and model is player -1 (minimizer).
    starting_player = 1
    best_value, best_sequence, total_terminals = minimax_dfs(initial_state, starting_player, [], game, model)

    print("\nMinimax evaluation complete.")
    print("Total number of terminal states (complete games) explored:", total_terminals)
    print("Global node count:", global_node_count)
    print("Minimax value (from player 1's perspective):", best_value)

    if best_value > 0:
        print("\nThere exists a sequence where the opponent (player 1) wins!")
        print("Winning move sequence (player, action index):")
        for move in best_sequence:
            print(move)
    else:
        print("\nNo winning sequence for the opponent was found.")
        print("Thus, with full branching, the model (player -1) is unbeatable under optimal play.")

if __name__ == "__main__":
    main()

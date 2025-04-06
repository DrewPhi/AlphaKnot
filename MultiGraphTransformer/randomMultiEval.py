#!/usr/bin/env python3
import os
import numpy as np
import random

from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from arena import Arena
from mcts import MCTS
from pd_code_utils import pd_code_from_graph
import config

def currentNetPlayer(game, nnet, board, player):
    canonicalBoard, current_player = game.getCanonicalForm(board, player)
    pi = MCTS(game, nnet, config, add_root_noise=False).getActionProb(canonicalBoard, current_player, temp=0)
    return int(np.argmax(pi))

def main():
    checkpoint_path = os.path.join(config.checkpoint, 'best.pth.tar')
    results = []
    
    # Iterate over each knot (each PD code) in config.pd_codes.
    for i, pd_code in enumerate(config.pd_codes):
        print(f"\n=== Evaluating Knot {i+1}/{len(config.pd_codes)} ===")
        # Create a new game instance.
        current_game = KnotGraphGame()
        
        # Define a fixed initialization function that always returns a board built from the current PD code.
        def fixed_init_board():
            current_game.pd_code = pd_code  # Assign the fixed PD code so that getActionSize() works.
            current_game.graph = current_game.pd_code_to_graph_data(pd_code)
            return current_game.graph
        current_game.getInitBoard = fixed_init_board
        
        # Call getInitBoard once to initialize current_game.pd_code.
        _ = current_game.getInitBoard()
        
        # Instantiate the neural net wrapper and load the checkpoint.
        nnet = NNetWrapper(current_game)
        if os.path.isfile(checkpoint_path):
            try:
                nnet.load_checkpoint(checkpoint_path)
                print(f"Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                print("Error loading checkpoint; using untrained model.", e)
        else:
            print("No checkpoint found; using current model weights (likely untrained).")
        
        # Define a random player function for this knot that uses the current game.
        def random_player(board, player):
            valid_moves = current_game.getValidMoves(board, player).cpu().numpy()
            valid_actions = np.where(valid_moves == 1)[0]
            return int(np.random.choice(valid_actions))
        
        # Scenario 1: AI as first (player 1) vs. Random (player -1)
        arena1 = Arena(
            lambda board, player: currentNetPlayer(current_game, nnet, board, player),
            random_player,
            current_game
        )
        oneWon, twoWon, draws = arena1.playGames(100, verbose=False)
        # In this arena, AI is player 1.
        ai_win_rate_first = (oneWon / 100) * 100
        
        # Scenario 2: AI as second (player -1) vs. Random (player 1)
        arena2 = Arena(
            random_player,
            lambda board, player: currentNetPlayer(current_game, nnet, board, player),
            current_game
        )
        oneWon2, twoWon2, draws2 = arena2.playGames(100, verbose=False)
        # Here, AI is player -1.
        ai_win_rate_second = (twoWon2 / 100) * 100
        
        print(f"Knot {i+1}:")
        print(f"  AI as FIRST player vs. Random: AI win rate = {ai_win_rate_first:.2f}%")
        print(f"  AI as SECOND player vs. Random: AI win rate = {ai_win_rate_second:.2f}%")
        
        results.append({
            'knot_index': i+1,
            'ai_first_winrate': ai_win_rate_first,
            'ai_second_winrate': ai_win_rate_second
        })
    
    print("\n=== Summary of Results ===")
    for r in results:
        print(f"Knot {r['knot_index']}: AI first win rate = {r['ai_first_winrate']:.2f}%, "
              f"AI second win rate = {r['ai_second_winrate']:.2f}%")
        
if __name__ == "__main__":
    main()

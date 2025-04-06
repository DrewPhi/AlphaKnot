#!/usr/bin/env python3
import numpy as np
import os
import random

from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from mcts import MCTS
from pd_code_utils import pd_code_from_graph
import config

# -------------------------------
# PARAMETERS
# -------------------------------
RANDOM_MODE = True       # True to replace human with a random player.
RANDOM_GAMES = 100       # Number of games to simulate when in RANDOM_MODE.
RANDOM_FIRST = 1         # Set to 1 if the random player goes first, or -1 if it goes second.
CHECKPOINT_PATH = './checkpoints/checkpoint_1.pth.tar'

# -------------------------------
# Player functions
# -------------------------------
def currentNetPlayer(game, nnet, board, player):
    canonicalBoard, current_player = game.getCanonicalForm(board, player)
    pi = MCTS(game, nnet, config, add_root_noise=False).getActionProb(canonicalBoard, current_player, temp=0)
    return np.argmax(pi)

def humanPlayer(game, board, player):
    valids = game.getValidMoves(board, player).cpu().numpy()
    valid_actions = np.where(valids == 1)[0]
    pd_code = pd_code_from_graph(board)

    print(f"\n🧠 Valid actions for Player {player}:")
    for action in valid_actions:
        crossing = action // 2
        resolution_type = action % 2  # 0 = one way, 1 = mirror resolution
        if crossing < len(pd_code):
            pd_entry = pd_code[crossing]
            print(f"{action}: Flip crossing {crossing} [type {resolution_type}] → {pd_entry}")

    while True:
        try:
            move = int(input("🎯 Enter your move: "))
            if move in valid_actions:
                return move
            else:
                print("⛔ Invalid move. Try again.")
        except:
            print("⚠️ Enter a valid integer.")

def randomPlayer(game, board, player):
    valid_moves_tensor = game.getValidMoves(board, player)
    valid_moves = valid_moves_tensor.cpu().numpy()
    valid_actions = np.where(valid_moves == 1)[0]
    return int(np.random.choice(valid_actions))

def print_pd_code(board, label=""):
    pd_code = pd_code_from_graph(board)
    print(f"\n🧵 PD Code {label}:")
    for i, crossing in enumerate(pd_code):
        print(f"  Crossing {i}: {crossing}")

# -------------------------------
# Interactive game (human vs AI)
# -------------------------------
def interactiveGame():
    game = KnotGraphGame()
    board = game.getInitBoard()
    curPlayer = 1

    print("🧩 Starting Game")
    print_pd_code(board, "at Start")

    # Load neural network
    nnet = NNetWrapper(game)
    nnet.load_checkpoint(CHECKPOINT_PATH)

    print("\n🤖 Who should go first?")
    print("1 = You, 2 = AI")
    first = input("Choice: ").strip()

    if first == "1":
        human_is = 1
    else:
        human_is = -1

    while True:
        print(f"\n=== Player {curPlayer}'s turn ===")
        if curPlayer == human_is:
            action = humanPlayer(game, board, curPlayer)
        else:
            action = currentNetPlayer(game, nnet, board, curPlayer)
            print(f"🤖 AI (Player {curPlayer}) chooses action: {action}")

        board, curPlayer = game.getNextState(board, curPlayer, action)
        print_pd_code(board, f"after move {action}")

        game_over = game.getGameEnded(board, curPlayer)
        if game_over != 0:
            print("\n Game Over!")
            if game_over == 1:
                print("Player 1 wins!")
            elif game_over == -1:
                print("Player -1 wins!")
            else:
                print("It's a draw!")
            if game_over == human_is:
                print('You win!')
            else:
                print('You lose!')
            print_pd_code(board, "at End")
            break

# -------------------------------
# Simulation mode (random vs AI)
# -------------------------------
def simulateRandomGames():
    game = KnotGraphGame()
    initial_state = game.getInitBoard()

    nnet = NNetWrapper(game)
    
    # Attempt to load a pre-trained model checkpoint.
    if os.path.isfile(CHECKPOINT_PATH):
        try:
            nnet.load_checkpoint(CHECKPOINT_PATH)
            print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
        except Exception as e:
            print("Error loading checkpoint; using untrained model.", e)
    else:
        print("No checkpoint found; using current model weights (likely untrained).")

    # In simulation, the random player will take the role normally held by the human.
    # We set 'human_is' equal to RANDOM_FIRST.
    human_is = RANDOM_FIRST  
    random_wins = 0

    for i in range(1, RANDOM_GAMES + 1):
        board = game.getInitBoard()
        curPlayer = 1
        while True:
            if curPlayer == human_is:
                action = randomPlayer(game, board, curPlayer)
            else:
                action = currentNetPlayer(game, nnet, board, curPlayer)
            board, curPlayer = game.getNextState(board, curPlayer, action)
            game_over = game.getGameEnded(board, curPlayer)
            if game_over != 0:
                # Outcome: game_over == 1 means Player 1 wins; -1 means Player -1 wins.
                if game_over == human_is:
                    random_wins += 1
                break
        print(f"Game {i} complete. Outcome: {game_over}")
    winrate = random_wins / RANDOM_GAMES * 100
    print(f"\nRandom player (Player {human_is}) win rate over {RANDOM_GAMES} games: {winrate:.2f}%")

# -------------------------------
# Main
# -------------------------------
def main():
    if RANDOM_MODE:
        simulateRandomGames()
    else:
        interactiveGame()

if __name__ == "__main__":
    main()

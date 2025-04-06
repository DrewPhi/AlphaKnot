# check_terminal_game_states.py

import torch
from knot_graph_game import KnotGraphGame
import config
import random

def fully_play_game():
    game = KnotGraphGame()
    board = game.getInitBoard()
    player = 1

    print("🧪 Starting full playthrough of a random game...")

    step = 0
    while True:
        valids = game.getValidMoves(board, player).cpu().numpy()
        valid_actions = [i for i, v in enumerate(valids) if v == 1]

        if not valid_actions:
            print(f"✅ No valid moves remain at step {step}")
            break

        action = random.choice(valid_actions)  # just pick the first valid move deterministically
        board, player = game.getNextState(board, player, action)
        step += 1

    print(f"🎯 Game reached terminal state in {step} moves")
    result = game.getGameEnded(board, player)

    if result == 0:
        print("❌ ERROR: getGameEnded returned 0 even after all moves were played.")
    elif result == 1:
        print("✅ Game ended: Current player wins.")
    elif result == -1:
        print("✅ Game ended: Opponent wins.")
    else:
        print(f"⚠️ Unexpected result: {result}")

if __name__ == "__main__":
    while True:
     fully_play_game()

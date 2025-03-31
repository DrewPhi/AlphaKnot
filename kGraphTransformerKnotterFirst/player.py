import numpy as np
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from mcts import MCTS
from pd_code_utils import pd_code_from_graph  # ✅ New PD code reconstruction
import config

CHECKPOINT_PATH = './checkpoints/best.pth.tar'

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
        resolution_type = action % 2  # 0 = one way, 1 = the mirror
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

def print_pd_code(board, label=""):
    pd_code = pd_code_from_graph(board)
    print(f"\n🧵 PD Code {label}:")
    for i, crossing in enumerate(pd_code):
        print(f"  Crossing {i}: {crossing}")

def main():
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
            print("\n🎉 Game Over!")
            if game_over == 1:
                print("🏆 Player 1 wins!")
            elif game_over == -1:
                print("🏆 Player -1 wins!")
            else:
                print("🤝 It's a draw!")

            print_pd_code(board, "at End")
            break

if __name__ == "__main__":
    main()

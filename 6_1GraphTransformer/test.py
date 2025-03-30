import numpy as np
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from mcts import MCTS
import config

class Args:
    numMCTSSims = config.numMCTSSims
    cpuct = config.cpuct

def random_player(game):
    def player_fn(board):
        valids = game.getValidMoves(board, 1).cpu().numpy()
        valid_actions = np.where(valids == 1)[0]
        return np.random.choice(valid_actions)
    return player_fn

def mcts_player(game, nnet, player_name="AI"):
    def player_fn(board):
        mcts = MCTS(game, nnet, Args())
        pi = mcts.getActionProb(board, temp=0)
        print(f"{player_name} policy π:", np.round(pi, 2))
        return np.argmax(pi)
    return player_fn

def run_test(game, ai_first=True):
    print("\n==============================")
    print("🔍 TEST: AI goes", "first" if ai_first else "second")
    print("==============================")

    nnet = NNetWrapper(game)
    board = game.getInitBoard()
    curPlayer = 1

    if ai_first:
        players = {1: mcts_player(game, nnet, "AI"), -1: random_player(game)}
    else:
        players = {1: random_player(game), -1: mcts_player(game, nnet, "AI")}

    move_num = 1
    while True:
        print(f"\n🔁 Move {move_num} by Player {curPlayer}")
        action = players[curPlayer](board)

        board, curPlayer = game.getNextState(board, curPlayer, action)

        last_player = -curPlayer  # the player who just moved
        result = game.getGameEnded(board, last_player)

        if result != 0:
            winner = "Player 1" if result * last_player == 1 else "Player 2"
            print("\n🏁 Game Over!")
            print(f"Final Player (who moved last): {last_player}")
            print(f"getGameEnded Result: {result}")
            print(f"Winner (from Arena perspective): {winner}")

            if (ai_first and winner == "Player 1") or (not ai_first and winner == "Player 2"):
                print("✅ AI won!")
            else:
                print("❌ AI lost!")
            break

        move_num += 1

if __name__ == "__main__":
    game = KnotGraphGame()
    game.getInitBoard()
    run_test(game, ai_first=True)
    run_test(game, ai_first=False)

from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from mcts import MCTS
import numpy as np
import torch

class DummyArgs:
    numMCTSSims = 25
    cpuct = 1.0
def test_get_game_ended():
    print("=== Testing getGameEnded ===")
    game = KnotGraphGame()
    board = game.getInitBoard()

    print("Initial PD Code:", game.pd_code)
    print("Initial edge signs (should be all 0):", board.edge_attr[:, 1].tolist())

    # Manually set all signs to simulate a finished game
    # We'll alternate signs for variety
    signs = [1 if i % 4 < 2 else 2 for i in range(board.edge_attr.shape[0])]
    board.edge_attr[:, 1] = torch.tensor(signs, dtype=torch.float)

    print("Manually set edge signs:", board.edge_attr[:, 1].tolist())

    result = game.getGameEnded(board, player=1)
    print("Game result (should be 1 or -1):", result)

    assert result in [1, -1], "Game should end when all signs are filled."
    print("✅ getGameEnded passes test.\n")

def test_mcts():
    print("=== MCTS Testing ===")

    game = KnotGraphGame()
    board = game.getInitBoard()
    print("Initial PD Code:", game.pd_code)
    print("Board shape:", board.x.shape)
    print("Edge index shape:", board.edge_index.shape)
    print("Edge attr shape:", board.edge_attr.shape)

    nnet = NNetWrapper(game)
    args = DummyArgs()
    mcts = MCTS(game, nnet, args)

    print("\n=== Running MCTS Simulations ===")
    probs = mcts.getActionProb(board, temp=1)
    print("Action probabilities:", np.round(probs, 3))
    print("Sum of probabilities:", np.sum(probs))

    print("\n=== Internal MCTS Data Inspection ===")
    s = game.stringRepresentation((board, 1))

    print(f"\nVisit count Ns[{s[:40]}...] = {mcts.Ns.get(s, 0)}")
    for a in range(game.getActionSize()):
        if (s, a) in mcts.Nsa:
            print(f"Action {a}: Nsa = {mcts.Nsa[(s, a)]}, Qsa = {round(mcts.Qsa[(s, a)], 3)}")

    print("\n=== Testing Greedy Selection (temp = 0) ===")
    greedy_probs = mcts.getActionProb(board, temp=0)
    best_move = np.argmax(greedy_probs)
    print("Greedy selected move:", best_move)
    print("Greedy probs:", greedy_probs)

    print("\n✅ MCTS test complete. All major functions triggered and inspected.")
from knot_graph_game import KnotGraphGame
import torch

def test_getNextState_progress():
    print("=== Testing getNextState() Progress ===")
    game = KnotGraphGame()
    board = game.getInitBoard()

    valid_moves = game.getValidMoves(board, 1)
    print("Valid moves at start:", valid_moves.nonzero().flatten().tolist())
    if not valid_moves.any():
        print("❌ No valid moves found.")
        return

    action = valid_moves.nonzero().flatten()[0].item()

    board2, next_player = game.getNextState(board, 1, action)
    print("Player after move:", next_player)

    # Check signs were updated
    signs1 = board.edge_attr[:, 1].tolist()
    signs2 = board2.edge_attr[:, 1].tolist()

    print("Edge signs before:", signs1)
    print("Edge signs after :", signs2)

    if signs1 == signs2:
        print("❌ No change in edge signs — game didn't progress.")
    else:
        print("✅ Edge signs updated.")

    valid_moves2 = game.getValidMoves(board2, next_player)
    if valid_moves2.sum() >= valid_moves.sum():
        print("❌ Valid move count did not decrease.")
    else:
        print("✅ Valid moves decreased after state transition.")

from knot_graph_game import KnotGraphGame
import torch
import random

def test_game_loop_until_end():
    print("=== Testing Full Game Loop ===")
    game = KnotGraphGame()
    board = game.getInitBoard()
    player = 1
    move_count = 0

    print("Initial PD Code:", game.pd_code)

    while True:
        ended = game.getGameEnded(board, player)
        if ended != 0:
            print(f"\n🏁 Game ended after {move_count} moves. Result: {ended}")
            break

        valid_moves = game.getValidMoves(board, player)
        valid_indices = valid_moves.nonzero().flatten().tolist()

        if not valid_indices:
            print("❌ No valid moves left but game not ended — possible logic error.")
            break

        action = random.choice(valid_indices)
        board, player = game.getNextState(board, player, action)

        move_count += 1
        print(f"Move {move_count}: player {player}, action {action}")
        print("Edge signs:", board.edge_attr[:, 1].tolist())

    print("✅ Game termination behavior appears correct.")



if __name__ == "__main__":
    test_mcts()
    #test_getNextState_progress()
    #test_get_game_ended()
    #test_game_loop_until_end()


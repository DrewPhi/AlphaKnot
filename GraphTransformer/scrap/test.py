import torch
from knot_graph_game import KnotGraphGame
from pd_code_utils import pd_code_from_graph, isPositive

def print_pd(pd, label):
    print(f"\n🔍 {label} PD Code:")
    for i, cr in enumerate(pd):
        print(f"  Crossing {i}: {cr}")

def test_all_actions():
    game = KnotGraphGame()
    base_board = game.getInitBoard()
    original_pd = pd_code_from_graph(base_board)

    print("🧪 Testing all 12 possible actions:")
    for action in range(12):
        crossing_index = action // 2
        resolution_type = action % 2  # 0 = even (type 0), 1 = odd (type 1)

        # Run action
        board, _ = game.getNextState(base_board, 1, action)
        pd = pd_code_from_graph(board)

        original_crossing = original_pd[crossing_index]
        new_crossing = pd[crossing_index]

        # Check if flipping behavior matches expected type
        flipped = new_crossing != original_crossing
        expected_flip = resolution_type == 1  # we expect type 1 to flip

        status = "✅ Correct" if flipped == expected_flip else "❌ Backwards"
        print(f"\n🔹 Action {action}: Crossing {crossing_index} [Type {resolution_type}] → {new_crossing}")
        print(f"    Original: {original_crossing}")
        print(f"    Flipped? {flipped} | Expected Flip? {expected_flip} → {status}")

if __name__ == "__main__":
    test_all_actions()

# test_pd_code_3_detailed_nodes.py

import random
from knot_graph_game import KnotGraphGame

def print_edge_details(board):
    """
    Prints detailed information for each edge:
      - Edge index
      - Source node (its actual feature / crossing)
      - Destination node (its actual feature / crossing)
      - Strand label
      - Current sign
    """
    edge_index = board.edge_index  # shape: [2, num_edges]
    edge_attr = board.edge_attr    # each row: [strand_label, sign]
    # Retrieve node features from board.x (each row corresponds to a crossing)
    nodes = board.x.tolist()
    num_edges = edge_index.shape[1]
    
    print("Edge Details:")
    for i in range(num_edges):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        # Look up the actual node features (the crossing PD code)
        src_node_feature = nodes[src_idx]
        dst_node_feature = nodes[dst_idx]
        strand, sign = edge_attr[i].tolist()
        print(f"  Edge {i}: Source {src_node_feature} -> Destination {dst_node_feature}, Strand {strand}, Sign {sign}")
    print()

def main():
    game = KnotGraphGame()
    # Use the PD code [[1,5,2,4],[3,1,4,6],[5,3,6,2]]
    board = game.getInitBoard()
    current_player = 1
    move_count = 0

    print("=== Starting Game with PD code of length 3 ===")
    print("Initial PD code:", game.initial_pd_code)
    print_edge_details(board)

    # Continue until game is over (i.e. getGameEnded returns nonzero)
    while True:
        result = game.getGameEnded(board, current_player)
        if result != 0:
            print("\nGame ended!")
            print("Final board state:")
            print_edge_details(board)
            break

        valid_moves = game.getValidMoves(board, current_player).tolist()
        print(f"Move {move_count}, Current player: {current_player}")
        print("Valid moves:", valid_moves)
        
        num_nodes = board.x.shape[0]
        # For each node, print its actual crossing (PD code) along with move validity.
        for node in range(num_nodes):
            crossing = board.x[node].tolist()
            pos_valid = bool(valid_moves[2 * node])
            neg_valid = bool(valid_moves[2 * node + 1])
            print(f"  Crossing {crossing}: Positive valid? {pos_valid}, Negative valid? {neg_valid}")
        
        possible_actions = [i for i, valid in enumerate(valid_moves) if valid == 1]
        if not possible_actions:
            print("No valid moves remain! (Draw state)")
            break

        action = random.choice(possible_actions)
        print(f"Chosen action: {action}\n")
        
        board, current_player = game.getNextState(board, current_player, action)
        print_edge_details(board)
        move_count += 1

if __name__ == "__main__":
    main()

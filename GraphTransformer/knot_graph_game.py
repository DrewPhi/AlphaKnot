# knot_graph_game.py

import random
import torch
from torch_geometric.data import Data
import config
import snappy
from pd_code_utils import pd_code_from_graph

class KnotGraphGame:
    def __init__(self):
        self.pd_code = None  # The current PD code
        self.graph = None    # The graph representation (Data object)
    @property
    def initial_pd_code(self):
        return self.pd_code

    def getInitBoard(self):
        self.pd_code = random.choice(config.pd_codes)
        self.graph = self.pd_code_to_graph_data(self.pd_code)
        return self.graph

    def pd_code_to_graph_data(self, pd_code):
        nodes = [tuple(crossing) for crossing in pd_code]
        node_map = {node: i for i, node in enumerate(nodes)}

        x = torch.tensor(pd_code, dtype=torch.float)

        edge_index = []
        edge_attr = []

        def connected_to(strand, others):
            for c in others:
                if strand in c:
                    return c
            return None

        for i in pd_code:
            i_tuple = tuple(i)
            trunc = [j for j in pd_code if j != i]
            c0 = connected_to(i[0], trunc)
            c2 = connected_to(i[2], trunc)
            if c0:
                edge_index.append([node_map[tuple(c0)], node_map[i_tuple]])
                edge_index.append([node_map[i_tuple], node_map[tuple(c0)]])
                edge_attr.append([i[0], 0])
                edge_attr.append([i[0], 0])
            if c2:
                edge_index.append([node_map[i_tuple], node_map[tuple(c2)]])
                edge_index.append([node_map[tuple(c2)], node_map[i_tuple]])
                edge_attr.append([i[2], 0])
                edge_attr.append([i[2], 0])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def getBoardSize(self):
        # (node feature dimension, max number of nodes)
        return (4, None)

    def getActionSize(self):
        # Each action is: choose a node, then apply one of 2 signs
        return 2 * len(self.pd_code)

        # Placeholder methods:
    def getNextState(self, board, player, action):
        new_board = Data(
            x=board.x.clone(),
            edge_index=board.edge_index.clone(),
            edge_attr=board.edge_attr.clone()
        )
        # Only use 1 = positive, 2 = negative
        SIGN_POSITIVE = 1
        SIGN_NEGATIVE = 2

        node_idx = action // 2
        sign_choice = SIGN_POSITIVE if action % 2 == 0 else SIGN_NEGATIVE
        alt_sign = SIGN_NEGATIVE if sign_choice == SIGN_POSITIVE else SIGN_POSITIVE

        # Step 1: Collect all outgoing edges from this node
        outgoing_edges = []
        for i in range(new_board.edge_index.shape[1]):
            src, dst = new_board.edge_index[:, i]
            if src.item() == node_idx:
                outgoing_edges.append((i, int(new_board.edge_attr[i][0])))  # (edge idx, strand label)
        # Step 2: Group strand labels into two consecutive pairs (with circular check)
        n_strands = int(new_board.edge_attr[:, 0].max().item())  # total strand labels
        strand_labels = sorted([strand for _, strand in outgoing_edges])
        # Use a greedy circular-adjacency pairing strategy
        grouped = []
        used = set()
        for i in range(len(strand_labels)):
            for j in range(i + 1, len(strand_labels)):
                a, b = strand_labels[i], strand_labels[j]
                if a in used or b in used:
                    continue
                diff = abs(a - b)
                if diff == 1 or diff == n_strands - 1:
                    grouped.append((a, b))
                    used.add(a)
                    used.add(b)
                    break

        if len(grouped) != 2:
            raise ValueError(f"Could not find two pairs of consecutive strands in: {strand_labels}")

        # Step 3: Assign signs
        pair1, pair2 = grouped
        # Step 3: Determine which group contains the first strand
        pd_entry = list(map(int, new_board.x[node_idx].tolist()))
        first_strand = pd_entry[0]

        if first_strand in pair1:
            primary, secondary = pair1, pair2
        else:
            primary, secondary = pair2, pair1

        # Step 4: Assign signs based on type
        if action % 2 != 0:  # even = primary gets understrand
            sign_map = {
                primary[0]: SIGN_NEGATIVE,
                primary[1]: SIGN_NEGATIVE,
                secondary[0]: SIGN_POSITIVE,
                secondary[1]: SIGN_POSITIVE,
            }
        else:  # odd = primary gets overstrand
            sign_map = {
                primary[0]: SIGN_POSITIVE,
                primary[1]: SIGN_POSITIVE,
                secondary[0]: SIGN_NEGATIVE,
                secondary[1]: SIGN_NEGATIVE,
            }


        # Step 4: Update edge_attr for outgoing edges
        for i, strand in outgoing_edges:
            new_sign = sign_map[strand]
            new_board.edge_attr[i][1] = new_sign

        return new_board, -1 * player  # Flip between 1 and -1


    def getValidMoves(self, board, player):
        num_nodes = board.x.shape[0]
        valid = [0] * (2 * num_nodes)

        # Check which nodes (crossings) still have unresolved signs
        for node_idx in range(num_nodes):
            # Find outgoing edges for this node
            outgoing_signs = []
            for i in range(board.edge_index.shape[1]):
                src, dst = board.edge_index[:, i]
                if src.item() == node_idx:
                    outgoing_signs.append(board.edge_attr[i][1].item())

            # If all signs are still 0, this node can be acted upon
            if all(sign == 0 for sign in outgoing_signs):
                valid[2 * node_idx] = 1     # +1 option
                valid[2 * node_idx + 1] = 1 # -1 option

        return torch.tensor(valid, dtype=torch.uint8)


    @staticmethod
    def reconstruct_pd_code(data: Data):
        return [list(map(int, node.tolist())) for node in data.x]


    def getGameEnded(self, board, player):
        # Not finished if any edge sign is unset
        if (board.edge_attr[:, 1] == 0).any():
            return 0  # game is still ongoing

        # Reconstruct PD code from the graph with resolved signs
        pd_code = pd_code_from_graph(board)

        try:
            link = snappy.Link(pd_code)

            # Quick check: if Alexander polynomial is not 1, it's knotted
            if str(link.alexander_polynomial()) != '1':
                is_unknot = False
            else:
                is_unknot = (str(link.jones_polynomial()) == '1')

        except Exception as e:
            print(f"SnapPy error: {e}")
            is_unknot = False  # Fallback to conservative outcome

        # Determine result based on which player is the unknotter
        unknotter = 1 if not config.knotter_first else -1

        if is_unknot:
            return 1 if player == unknotter else -1
        else:
            return 1 if player != unknotter else -1



    def getCanonicalForm(self, board, player):
        return board, player  # return both



    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board_player_tuple):
        board, player = board_player_tuple
        # Include edge signs so that different resolution states are distinct
        sign_list = board.edge_attr[:, 1].tolist()
        return str(board.x.tolist()) + "_" + str(sign_list) + f"_p{player}"





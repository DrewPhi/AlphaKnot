# knot_graph_game.py

import random
import torch
from torch_geometric.data import Data
import config
from pd_code_utils import canonicalize_pd_code, pd_code_from_graph
from knot_invariants import jones_is_one

class KnotGraphGame:
    def __init__(self, pd_code=None):
        self._fixed_pd_code = (
            [list(map(int, crossing)) for crossing in pd_code]
            if pd_code is not None
            else None
        )
        self.pd_code = None  # The current PD code
        self.source_pd_code = None
        self.pd_canonicalization = None
        self.graph = None    # The graph representation (Data object)
    @property
    def initial_pd_code(self):
        return self.pd_code

    def getInitBoard(self):
        source = (
            self._fixed_pd_code
            if self._fixed_pd_code is not None
            else random.choice(config.pd_codes)
        )
        self.source_pd_code = [list(crossing) for crossing in source]
        self.pd_canonicalization = canonicalize_pd_code(self.source_pd_code)
        self.pd_code = self.pd_canonicalization.as_lists()
        self.graph = self.pd_code_to_graph_data(self.pd_code)
        return self.graph

    def canonical_action_to_source(self, action):
        return self.pd_canonicalization.canonical_action_to_source(action)

    def source_action_to_canonical(self, action):
        return self.pd_canonicalization.source_action_to_canonical(action)

    def pd_code_to_graph_data(self, pd_code):
        pd_code = canonicalize_pd_code(pd_code).as_lists()
        nodes = [tuple(crossing) for crossing in pd_code]
        node_map = {node: i for i, node in enumerate(nodes)}

        # Node columns 0:4 are the ordered PD tuple.  Column 4 is the crossing
        # resolution state: 0 unresolved, 1 original, 2 switched.
        x = torch.tensor(
            [list(crossing) + [0] for crossing in pd_code],
            dtype=torch.float,
        )

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
        return (5, None)

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
        # Edge attributes describe which strand passes under/over.  They are
        # not crossing writhe signs.
        STRAND_UNDER = 1
        STRAND_OVER = 2

        node_idx = action // 2
        # Collect all outgoing edges from this node.
        outgoing_edges = []
        for i in range(new_board.edge_index.shape[1]):
            src, dst = new_board.edge_index[:, i]
            if src.item() == node_idx:
                outgoing_edges.append((i, int(new_board.edge_attr[i][0])))  # (edge idx, strand label)
        # PD tuple position defines the strands: (a, c) is under and (b, d)
        # is over.  Do not reconstruct this information by sorting labels.
        pd_entry = list(map(int, new_board.x[node_idx, :4].tolist()))
        a, b, c, d = pd_entry
        original_under = (a, c)
        original_over = (b, d)

        if action % 2 == 0:  # keep the original crossing
            new_board.x[node_idx, 4] = 1
            sign_map = {
                a: STRAND_UNDER,
                c: STRAND_UNDER,
                b: STRAND_OVER,
                d: STRAND_OVER,
            }
        else:  # crossing change
            new_board.x[node_idx, 4] = 2
            sign_map = {
                a: STRAND_OVER,
                c: STRAND_OVER,
                b: STRAND_UNDER,
                d: STRAND_UNDER,
            }

        outgoing_labels = {strand for _, strand in outgoing_edges}
        expected_labels = set(original_under + original_over)
        if outgoing_labels != expected_labels:
            raise ValueError(
                f"Graph/PD mismatch at node {node_idx}: "
                f"expected {expected_labels}, got {outgoing_labels}"
            )

        # Update edge attributes for outgoing edges.
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
        return [list(map(int, node[:4].tolist())) for node in data.x]


    def getGameEnded(self, board, player):
        # Not finished if any edge sign is unset
        if (board.edge_attr[:, 1] == 0).any():
            return 0  # game is still ongoing
        # Reconstruct PD code from the graph with resolved signs
        pd_code = pd_code_from_graph(board)
        if len(pd_code) > config.max_validated_crossings:
            raise ValueError(
                f"Refusing to classify {len(pd_code)} crossings; "
                f"validated maximum is {config.max_validated_crossings}"
            )

        try:
            is_unknot = jones_is_one(pd_code)
        except Exception as exc:
            raise RuntimeError(
                f"Could not classify terminal PD code {pd_code!r}"
            ) from exc

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

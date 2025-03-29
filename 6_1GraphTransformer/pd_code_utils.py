import torch
from torch_geometric.data import Data

def graph_to_dict(graph_data):
    """
    Converts a PyTorch Geometric Data object into a dictionary with:
    - 'nodes': list of PD code tuples
    - 'edges': list of (source_node, target_node, strand_label, sign)
    """
    nodes = [tuple(map(int, node.tolist())) for node in graph_data.x]
    idx_to_node = {idx: node for idx, node in enumerate(nodes)}
    
    edges = []
    for i in range(graph_data.edge_index.shape[1]):
        src_idx = int(graph_data.edge_index[0, i].item())
        dst_idx = int(graph_data.edge_index[1, i].item())
        strand_label = int(graph_data.edge_attr[i][0].item())
        sign = int(graph_data.edge_attr[i][1].item())

        src_node = idx_to_node[src_idx]
        dst_node = idx_to_node[dst_idx]

        edges.append((src_node, dst_node, strand_label, sign))

    return {
        "nodes": nodes,
        "edges": edges
    }

def print_graph_dict(graph_dict):
    print("=== Nodes ===")
    for node in graph_dict["nodes"]:
        print(f"  {node}")
    print("\n=== Edges ===")
    for src, dst, strand_label, sign in graph_dict["edges"]:
        print(f"  {src} → {dst}  (Strand: {strand_label}, Sign: {sign})")

def isPositive(crossing):
    """
    Determines whether a crossing is positive.
    """
    return crossing[1] > crossing[3]

def flip_crossing(crossing):
    """
    Flips a crossing to change its handedness.
    """
    if isPositive(crossing):
        return [crossing[3], crossing[0], crossing[1], crossing[2]]
    else:
        return [crossing[1], crossing[2], crossing[3], crossing[0]]

def pd_code_from_graph(graph_data):
    """
    Reconstruct the PD code from a graph, flipping crossings when needed
    based on the sign of the incoming edge for the first strand.
    """
    graph_dict = graph_to_dict(graph_data)
    new_pd_code = []

    for crossing in graph_dict["nodes"]:
        first_strand = crossing[0]
        needs_flip = False

        for src, dst, strand_label, sign in graph_dict["edges"]:
            if dst == crossing and strand_label == first_strand:
                if sign == 2:           # flip only for positive sign
                    needs_flip = True
                break


        if needs_flip:
            new_pd_code.append(flip_crossing(crossing))
        else:
            new_pd_code.append(list(crossing))

    return new_pd_code

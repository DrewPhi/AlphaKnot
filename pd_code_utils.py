import torch
from torch_geometric.data import Data


def serialize_graph(graph_data):
    """Convert a graph to plain lists safe for multiprocessing/pickling."""
    return {
        "x": graph_data.x.cpu().tolist(),
        "edge_index": graph_data.edge_index.cpu().tolist(),
        "edge_attr": graph_data.edge_attr.cpu().tolist(),
    }


def deserialize_graph(payload):
    """Reconstruct a PyG graph from :func:`serialize_graph` output."""
    required = {"x", "edge_index", "edge_attr"}
    if set(payload) != required:
        raise ValueError(f"Invalid serialized graph keys: {set(payload)}")
    return Data(
        x=torch.tensor(payload["x"], dtype=torch.float),
        edge_index=torch.tensor(payload["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(payload["edge_attr"], dtype=torch.float),
    )

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

def crossing_sign(crossing, n_edges):
    """Return ``+1`` or ``-1`` using the oriented, cyclic PD convention.

    AlphaKnot currently supports one-component PD codes whose edge labels are
    the consecutive integers ``1..n_edges``.  In ``[a, b, c, d]`` the upper
    strand is positive when it is oriented ``d -> b``.
    """
    if len(crossing) != 4:
        raise ValueError(f"Expected four PD labels, got {crossing!r}")
    if n_edges < 1:
        raise ValueError("n_edges must be positive")

    _, b, _, d = map(int, crossing)
    next_b = (b % n_edges) + 1
    next_d = (d % n_edges) + 1

    if b == next_d:
        return 1
    if d == next_b:
        return -1
    raise ValueError(
        "PD crossing does not use consecutive oriented upper-strand labels: "
        f"{crossing!r} (n_edges={n_edges})"
    )

def flip_crossing(crossing, n_edges):
    """Change the crossing while preserving the oriented PD convention."""
    if crossing_sign(crossing, n_edges) > 0:
        return [crossing[3], crossing[0], crossing[1], crossing[2]]
    return [crossing[1], crossing[2], crossing[3], crossing[0]]

def pd_code_from_graph(graph_data):
    """
    Reconstruct the PD code from a graph, flipping crossings when needed
    based on the sign of the incoming edge for the first strand.
    """
    graph_dict = graph_to_dict(graph_data)
    new_pd_code = []
    n_edges = max(label for crossing in graph_dict["nodes"] for label in crossing)

    for crossing in graph_dict["nodes"]:
        first_strand = crossing[0]
        needs_flip = False

        for src, dst, strand_label, sign in graph_dict["edges"]:
            if src == crossing and strand_label == first_strand:
                if sign == 2:  # the original under-strand was selected as over
                    needs_flip = True
                break


        if needs_flip:
            new_pd_code.append(flip_crossing(crossing, n_edges))
        else:
            new_pd_code.append(list(crossing))

    return new_pd_code

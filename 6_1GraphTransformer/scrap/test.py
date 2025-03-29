import torch
from torch_geometric.data import Data

def connected_to(strand, trunc):
    for i in trunc:
        if strand in i:
            return i
    return None

def pd_code_to_graph_data(pd_code):
    nodes = [tuple(crossing) for crossing in pd_code]
    node_map = {node: i for i, node in enumerate(nodes)}

    x = torch.tensor(pd_code, dtype=torch.float)
    edge_index = []
    edge_attr = []

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

def graph_to_dict(graph_data):
    """
    Converts a PyTorch Geometric Data object into a dictionary with:
    - 'nodes': list of PD code tuples
    - 'edges': list of (source_node, target_node, strand_label, sign)
    """

    # Convert node features back into PD code tuples
    nodes = [tuple(map(int, node.tolist())) for node in graph_data.x]
    
    # Create a reverse lookup for node index → PD code tuple
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



pd_code = [[1, 5, 2, 4], [3, 1, 4, 6], [5, 3, 6, 2]]
graph_data = pd_code_to_graph_data(pd_code)

print(graph_data)
print_graph_dict(graph_to_dict(graph_data))

def isPositive(crossing):
    #Given a crossing of the planar diagram code this outputs wether it is a positive or negative crossing.
    return crossing[1]>crossing[3]

def flip_crossing(crossing):
    if isPositive(crossing):
        return[crossing[3],crossing[0],crossing[1],crossing[2]]
    else:
        return[crossing[1],crossing[2],crossing[3],crossing[0]]

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

        # Look for the edge coming into this crossing with label == first_strand
        for src, dst, strand_label, sign in graph_dict["edges"]:
            if dst == crossing and strand_label == first_strand:
                if sign > 0:  # Needs flipping
                    needs_flip = True
                break

        if needs_flip:
            new_pd_code.append(flip_crossing(crossing))
        else:
            new_pd_code.append(list(crossing))

    return new_pd_code



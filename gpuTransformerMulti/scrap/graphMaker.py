import torch
from torch_geometric.data import Data
import snappy 

pd_code_3 = 	[[1,7,2,6],[3,10,4,11],[5,3,6,2],[7,1,8,12],[9,4,10,5],[11,9,12,8]]

def connected_to(strand,trunc):
    for i in trunc:
        if strand in i:
            return i
    return

def edge_finder(pd_code):
    edges = []
    for i in pd_code:
        i_tuple = tuple(i)
        trunc = [j for j in pd_code if j != i]
        connected0 = tuple(connected_to(i[0], trunc))
        connected2 = tuple(connected_to(i[2], trunc))

        edges.append((connected0, i_tuple, i[0]))
        edges.append((i_tuple, connected0, i[0]))
        edges.append((i_tuple, connected2, i[2]))
        edges.append((connected2, i_tuple, i[2]))
    return edges

print(edge_finder(pd_code_3))
    


# Store as a dictionary (clean and easy to access)
graph_set = {
    "nodes": [tuple(c) for c in pd_code_3],
    "edges": edge_finder(pd_code_3)
}

# Output
print("Nodes:")
for node in graph_set["nodes"]:
    print(" ", node)

print("\nEdges:")
for src, dst,lab in graph_set["edges"]:
    print(" ", src, "→", dst,lab)


def pd_code_to_graph_data(graph_set):
    # Map node tuples to integer indices
    node_map = {node: idx for idx, node in enumerate(graph_set["nodes"])}

    # Create node features: each node is a 4D vector (the PD code)
    x = torch.tensor(graph_set["nodes"], dtype=torch.float)

    # Prepare edges
    edge_index = []
    edge_attr = []

    for src, dst, strand_label in graph_set["edges"]:
        src_idx = node_map[src]
        dst_idx = node_map[dst]
        edge_index.append([src_idx, dst_idx])
        edge_attr.append([strand_label, 0])  # 0 is placeholder for sign

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # shape [num_edges, 2]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def inspect_graph_data(graph_data):
    print("=== Nodes ===")
    for i, features in enumerate(graph_data.x):
        pd_entry = [int(val.item()) for val in features]
        print(f"Node {i}: PD code {pd_entry}")

    print("\n=== Edges ===")
    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr

    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        strand_label, sign = edge_attr[i].tolist()
        print(f"Edge {i}: {src} → {dst} | Strand: {int(strand_label)}, Sign: {int(sign)}")
graph_data = pd_code_to_graph_data(graph_set)
inspect_graph_data(graph_data)




def reconstruct_pd_code(data: Data):
    return [list(map(int, node.tolist())) for node in data.x]


print(reconstruct_pd_code(graph_data))

print(snappy.Link(reconstruct_pd_code(graph_data)).jones_polynomial())
print(snappy.Link(reconstruct_pd_code(graph_data)).alexander_polynomial())
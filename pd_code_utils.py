from collections import Counter
from dataclasses import dataclass

import torch
from torch_geometric.data import Data


PD_CANONICALIZATION_VERSION = "oriented-dihedral-v1"
GRAPH_REPRESENTATION_VERSION = "pd-arc-incidence-v2"


@dataclass(frozen=True)
class CanonicalPDCode:
    """Canonical PD code and the crossing-coordinate map to its source."""

    pd_code: tuple
    canonical_to_source: tuple
    source_to_canonical: tuple
    orientation_reversed: bool
    label_offset: int

    def as_lists(self):
        return [list(crossing) for crossing in self.pd_code]

    def canonical_action_to_source(self, action):
        crossing, choice = divmod(int(action), 2)
        return 2 * self.canonical_to_source[crossing] + choice

    def source_action_to_canonical(self, action):
        crossing, choice = divmod(int(action), 2)
        return 2 * self.source_to_canonical[crossing] + choice


def _validate_oriented_pd_code(pd_code):
    if not pd_code:
        raise ValueError("PD code must contain at least one crossing")
    if any(len(crossing) != 4 for crossing in pd_code):
        raise ValueError("Every PD crossing must contain four labels")
    crossings = len(pd_code)
    n_edges = 2 * crossings
    labels = [int(label) for crossing in pd_code for label in crossing]
    occurrences = Counter(labels)
    if set(occurrences) != set(range(1, n_edges + 1)):
        raise ValueError(f"PD labels must be consecutive from 1 through {n_edges}")
    if any(count != 2 for count in occurrences.values()):
        raise ValueError("Every PD arc label must occur exactly twice")

    def cyclic_neighbors(left, right):
        return (left % n_edges) + 1 == right or (right % n_edges) + 1 == left

    for crossing in pd_code:
        a, b, c, d = map(int, crossing)
        if not cyclic_neighbors(a, c):
            raise ValueError(
                f"PD under-strand labels are not cyclic neighbors: {crossing!r}"
            )
        crossing_sign(crossing, n_edges)
    return n_edges


def canonicalize_pd_code(pd_code):
    """Canonicalize an oriented one-component PD serialization.

    The quotient includes cyclic traversal basepoint, component orientation,
    and crossing-list order.  Reversing orientation rotates every tuple by two
    slots, preserving the original-versus-switched action semantics.
    """
    source = tuple(tuple(map(int, crossing)) for crossing in pd_code)
    n_edges = _validate_oriented_pd_code(source)
    candidates = []
    for reverse in (False, True):
        for offset in range(n_edges):
            transformed = []
            for source_index, crossing in enumerate(source):
                labels = []
                for label in crossing:
                    coordinate = label - 1
                    if reverse:
                        coordinate = -coordinate
                    labels.append(((coordinate + offset) % n_edges) + 1)
                if reverse:
                    labels = labels[2:] + labels[:2]
                transformed.append((tuple(labels), source_index))
            transformed.sort()
            code = tuple(crossing for crossing, _ in transformed)
            canonical_to_source = tuple(index for _, index in transformed)
            candidates.append(
                (code, canonical_to_source, reverse, offset)
            )

    code, canonical_to_source, reverse, offset = min(candidates)
    source_to_canonical = [0] * len(source)
    for canonical_index, source_index in enumerate(canonical_to_source):
        source_to_canonical[source_index] = canonical_index
    return CanonicalPDCode(
        pd_code=code,
        canonical_to_source=canonical_to_source,
        source_to_canonical=tuple(source_to_canonical),
        orientation_reversed=reverse,
        label_offset=offset,
    )


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
    nodes = [tuple(map(int, node[:4].tolist())) for node in graph_data.x]
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

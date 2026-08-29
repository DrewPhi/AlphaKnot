#!/usr/bin/env python3
"""Exhaustively audit a checkpoint under equivalent PD serializations."""

import argparse

import torch
from torch_geometric.loader import DataLoader

from capacity_test import build_exact_dataset, evaluate
from exact_solver import ExactSolver
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from seven_crossing_corpus import corpus_records


def relabel_arcs(pd_code, offset=0, reverse=False):
    """Apply a dihedral relabeling of the oriented cyclic arc labels."""
    n_edges = max(label for crossing in pd_code for label in crossing)

    def transform(label):
        coordinate = int(label) - 1
        if reverse:
            coordinate = -coordinate
        return ((coordinate + offset) % n_edges) + 1

    relabeled = [[transform(label) for label in crossing] for crossing in pd_code]
    if reverse:
        # Reversing the component swaps incoming and outgoing half-edges on
        # both strands.  Rotate by two slots to keep AlphaKnot's oriented PD
        # convention and preserve keep/switch semantics.
        relabeled = [crossing[2:] + crossing[:2] for crossing in relabeled]
    return relabeled


def permute_crossings(pd_code, permutation):
    return [[int(label) for label in pd_code[index]] for index in permutation]


def equivalent_variants(pd_code):
    """Return representative basepoint, orientation, and list-order variants."""
    return {
        "original": [list(crossing) for crossing in pd_code],
        "basepoint+1": relabel_arcs(pd_code, offset=1),
        "basepoint+7": relabel_arcs(pd_code, offset=7),
        "orientation-reversed": relabel_arcs(pd_code, reverse=True),
        "crossings-reversed": permute_crossings(pd_code, tuple(range(6, -1, -1))),
        "crossings-shuffled": permute_crossings(pd_code, (2, 0, 6, 1, 5, 3, 4)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "shared_sweep_canonical_v1_t128_l6_b64_lr1e3.pth.tar"
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = corpus_records()
    first_game = KnotGraphGame(pd_code=records[0][1])
    first_game.getInitBoard()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    network = NNetWrapper(
        first_game,
        hidden_dim=args.hidden_dim,
        dropout=0.0,
        device=str(device),
        architecture="port-transformer-pd-position",
    )
    network.load_checkpoint(args.checkpoint, load_optimizer=False)

    total_states = 0
    total_policy = 0
    total_value = 0
    failures = []
    print("source variant states policy value optimal_mass solved")
    for source_name, source_pd in records:
        for variant_name, pd_code in equivalent_variants(source_pd).items():
            game = KnotGraphGame(pd_code=pd_code)
            game.getInitBoard()
            solver = ExactSolver(pd_code)
            dataset = build_exact_dataset(game, solver)
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False
            )
            metrics = evaluate(network.model, loader, device)
            states = metrics["states"]
            solved = (
                metrics["policy_correct"] == states
                and metrics["value_correct"] == states
            )
            print(
                f"{source_name:>4} {variant_name:>20} {states:5d} "
                f"{metrics['policy_correct']/states:8.2%} "
                f"{metrics['value_correct']/states:8.2%} "
                f"{metrics['optimal_mass']:12.2%} "
                f"{'YES' if solved else 'NO'}"
            )
            total_states += states
            total_policy += metrics["policy_correct"]
            total_value += metrics["value_correct"]
            if not solved:
                failures.append((source_name, variant_name))

    print(
        f"aggregate states={total_states} "
        f"policy={total_policy/total_states:.2%} "
        f"value={total_value/total_states:.2%}"
    )
    print(f"serialization_failures={len(failures)}")
    for source_name, variant_name in failures:
        print(f"FAILED {source_name} {variant_name}")
    print(f"EQUIVALENT_PD_INVARIANT: {'YES' if not failures else 'NO'}")


if __name__ == "__main__":
    main()

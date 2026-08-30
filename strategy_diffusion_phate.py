#!/usr/bin/env python3
"""Export matched strategy activations, diffusion operators, and PHATE maps."""

import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from evaluate_exact import board_from_state
from exact_solver import ExactSolver, UNRESOLVED
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from pd_code_utils import GRAPH_REPRESENTATION_VERSION, PD_CANONICALIZATION_VERSION
from prime_knot_corpus import corpus_records


PROBE_PANEL_VERSION = "repeat-all-n3-nonterminal-v1"
KERNEL_VERSION = "self-tuning-gaussian-knn5-v1"


def base_probe_states():
    """Return all 19 nonterminal ternary states on three crossings."""
    return tuple(
        state
        for state in product((UNRESOLVED, 0, 1), repeat=3)
        if UNRESOLVED in state
    )


def matched_probe_states(crossings):
    """Lift the fixed 3-crossing panel periodically to any larger diagram."""
    if crossings < 3:
        raise ValueError("the matched panel requires at least three crossings")
    states = tuple(
        tuple(template[index % 3] for index in range(crossings))
        for template in base_probe_states()
    )
    if len(set(states)) != len(states) or any(UNRESOLVED not in state for state in states):
        raise RuntimeError("matched probe construction must be unique and nonterminal")
    return states


def diffusion_operator(activations, knn=5):
    """Construct a self-tuning Gaussian affinity and row-normalize it."""
    activations = np.asarray(activations, dtype=np.float64)
    if activations.ndim != 2 or activations.shape[0] < 2:
        raise ValueError("activations must be a 2D matrix with at least two rows")
    differences = activations[:, None, :] - activations[None, :, :]
    squared_distances = np.einsum("ijk,ijk->ij", differences, differences)
    neighbor = min(max(int(knn), 1), activations.shape[0] - 1)
    distances = np.sqrt(np.maximum(squared_distances, 0.0))
    bandwidth = np.partition(distances, neighbor, axis=1)[:, neighbor]
    positive = bandwidth[bandwidth > 0]
    fallback = float(np.median(positive)) if positive.size else 1.0
    bandwidth = np.where(bandwidth > 0, bandwidth, fallback)
    scale = np.outer(bandwidth, bandwidth)
    affinity = np.exp(-squared_distances / np.maximum(scale, np.finfo(float).eps))
    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 1.0)
    return affinity / affinity.sum(axis=1, keepdims=True)


def pairwise_frobenius(matrices):
    flat = np.asarray(matrices, dtype=np.float64).reshape(len(matrices), -1)
    differences = flat[:, None, :] - flat[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", differences, differences))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_sha256(panel_by_size):
    payload = json.dumps(panel_by_size, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@torch.no_grad()
def extract_knot_activations(model, game, states, batch_size, device):
    graphs = [board_from_state(game, state) for state in states]
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    collected = {"value_penultimate": [], "graph_token": [], "crossing_mean": []}
    for batch in loader:
        batch = batch.to(device)
        _, _, activations = model(batch, return_activations=True)
        crossing_mask = activations["crossing_mask"].unsqueeze(-1)
        crossing_mean = (
            activations["crossing_tokens"] * crossing_mask
        ).sum(dim=1) / crossing_mask.sum(dim=1).clamp_min(1)
        collected["value_penultimate"].append(
            activations["value_penultimate"].cpu().numpy()
        )
        collected["graph_token"].append(activations["graph_token"].cpu().numpy())
        collected["crossing_mean"].append(crossing_mean.cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in collected.items()}


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output_dir, knot_embedding, state_embedding, knot_rows, state_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    crossings = np.array([row["crossings"] for row in knot_rows])
    figure, axis = plt.subplots(figsize=(9, 7))
    points = axis.scatter(
        knot_embedding[:, 0], knot_embedding[:, 1], c=crossings, cmap="viridis", s=65
    )
    for coordinates, row in zip(knot_embedding, knot_rows):
        axis.annotate(row["knot"], coordinates, fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.set(title="Prime-knot strategy diffusion operators", xlabel="PHATE 1", ylabel="PHATE 2")
    figure.colorbar(points, ax=axis, label="Crossing number")
    figure.tight_layout()
    figure.savefig(output_dir / "knot_operator_phate.png", dpi=220)
    figure.savefig(output_dir / "knot_operator_phate.svg")
    plt.close(figure)

    state_crossings = np.array([row["crossings"] for row in state_rows])
    figure, axis = plt.subplots(figsize=(9, 7))
    points = axis.scatter(
        state_embedding[:, 0], state_embedding[:, 1], c=state_crossings,
        cmap="viridis", s=12, alpha=0.75,
    )
    axis.set(title="Matched value-penultimate strategy states", xlabel="PHATE 1", ylabel="PHATE 2")
    figure.colorbar(points, ax=axis, label="Crossing number")
    figure.tight_layout()
    figure.savefig(output_dir / "state_activation_phate.png", dpi=220)
    figure.savefig(output_dir / "state_activation_phate.svg")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/strategy_phate_prime3to8")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kernel-knn", type=int, default=5)
    parser.add_argument("--phate-knn", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        import phate
    except ImportError as exc:
        raise SystemExit("Install requirements-analysis.txt before running PHATE") from exc

    records = corpus_records(3, 8)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    first_game = KnotGraphGame(pd_code=records[0][1])
    first_game.getInitBoard()
    network = NNetWrapper(
        first_game,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.0,
        device=str(device),
        architecture="variable-port-transformer",
    )
    network.load_checkpoint(args.checkpoint, load_optimizer=False)
    network.model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_activations = []
    all_graph_tokens = []
    all_crossing_means = []
    operators = []
    knot_rows = []
    state_rows = []
    panel_by_size = {
        str(crossings): matched_probe_states(crossings)
        for crossings in range(3, 9)
    }

    for knot_index, (name, pd_code) in enumerate(records):
        game = KnotGraphGame(pd_code=pd_code)
        game.getInitBoard()
        states = matched_probe_states(len(pd_code))
        extracted = extract_knot_activations(
            network.model, game, states, args.batch_size, device
        )
        primary = extracted["value_penultimate"]
        operator = diffusion_operator(primary, args.kernel_knn)
        all_activations.append(primary)
        all_graph_tokens.append(extracted["graph_token"])
        all_crossing_means.append(extracted["crossing_mean"])
        operators.append(operator)
        knot_rows.append(
            {"knot": name, "knot_index": knot_index, "crossings": len(pd_code)}
        )
        solver = ExactSolver(pd_code)
        for probe_index, state in enumerate(states):
            state_rows.append(
                {
                    "knot": name,
                    "knot_index": knot_index,
                    "crossings": len(pd_code),
                    "probe_index": probe_index,
                    "depth": sum(value != UNRESOLVED for value in state),
                    "player": solver.player_to_move(state),
                    "exact_value": solver.value(state),
                    "optimal_actions": len(solver.optimal_actions(state)),
                    "state": ";".join(map(str, state)),
                }
            )
        print(f"Extracted {name}: {primary.shape}")

    activations = np.stack(all_activations)
    graph_tokens = np.stack(all_graph_tokens)
    crossing_means = np.stack(all_crossing_means)
    operators = np.stack(operators)
    operator_distances = pairwise_frobenius(operators)
    knot_phate = phate.PHATE(
        n_components=2,
        knn=args.phate_knn,
        decay=40,
        t="auto",
        knn_dist="precomputed",
        random_state=args.seed,
        n_jobs=1,
        verbose=1,
    ).fit_transform(operator_distances)
    state_phate = phate.PHATE(
        n_components=2,
        knn=args.phate_knn,
        decay=40,
        t="auto",
        random_state=args.seed,
        n_jobs=1,
        verbose=1,
    ).fit_transform(activations.reshape(-1, activations.shape[-1]))

    np.savez_compressed(
        output_dir / "strategy_diffusion_data.npz",
        value_penultimate=activations.astype(np.float32),
        graph_token=graph_tokens.astype(np.float32),
        crossing_mean=crossing_means.astype(np.float32),
        diffusion_operators=operators.astype(np.float32),
        operator_distances=operator_distances.astype(np.float32),
        knot_phate=knot_phate.astype(np.float32),
        state_phate=state_phate.astype(np.float32),
    )
    write_csv(
        output_dir / "knot_operator_phate.csv",
        ["knot", "knot_index", "crossings", "phate_1", "phate_2"],
        [
            {**row, "phate_1": coordinates[0], "phate_2": coordinates[1]}
            for row, coordinates in zip(knot_rows, knot_phate)
        ],
    )
    write_csv(
        output_dir / "state_activation_phate.csv",
        list(state_rows[0]) + ["phate_1", "phate_2"],
        [
            {**row, "phate_1": coordinates[0], "phate_2": coordinates[1]}
            for row, coordinates in zip(state_rows, state_phate)
        ],
    )
    save_plots(output_dir, knot_phate, state_phate, knot_rows, state_rows)
    manifest = {
        "model_commit": "4a163c1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "architecture": "variable-port-transformer",
        "hidden_dim": args.hidden_dim,
        "activation": "value_penultimate",
        "activation_dimension": int(activations.shape[-1]),
        "knots": len(records),
        "probes_per_knot": int(activations.shape[1]),
        "probe_panel_version": PROBE_PANEL_VERSION,
        "probe_panel_sha256": panel_sha256(panel_by_size),
        "kernel_version": KERNEL_VERSION,
        "kernel_knn": args.kernel_knn,
        "phate_knn": args.phate_knn,
        "pd_canonicalization_version": PD_CANONICALIZATION_VERSION,
        "graph_representation_version": GRAPH_REPRESENTATION_VERSION,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "phate_version": getattr(phate, "__version__", "unknown"),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote strategy diffusion analysis to {output_dir}")


if __name__ == "__main__":
    main()

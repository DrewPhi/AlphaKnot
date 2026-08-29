#!/usr/bin/env python3
"""Test whether the current network can fit the complete seven-crossing table.

This is deliberately a capacity/memorization experiment, not an AlphaZero
self-play result. Every nonterminal state is used for both training and exact
evaluation.
"""

import argparse
from collections import Counter
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import config
from evaluate_exact import board_from_state
from exact_solver import ExactSolver, UNRESOLVED
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper


def exact_example(game, solver, state, depth_counts=None):
    """Build one graph with exact policy/value supervision attached."""
    board = board_from_state(game, state)
    action_size = game.getActionSize()
    legal = torch.zeros(action_size, dtype=torch.bool)
    legal[list(solver.legal_actions(state))] = True
    optimal = solver.optimal_actions(state)
    policy = torch.zeros(action_size, dtype=torch.float32)
    policy[list(optimal)] = 1.0 / len(optimal)
    depth = sum(choice != UNRESOLVED for choice in state)
    weight = 1.0 if depth_counts is None else 1.0 / depth_counts[depth]

    # A leading dimension makes PyG collate these as graph-level tensors.
    board.target_policy = policy.unsqueeze(0)
    board.target_value = torch.tensor([float(solver.value(state))])
    board.legal_mask = legal.unsqueeze(0)
    # Node-level copies collate across mixed crossing counts.  The legacy
    # graph-level tensors above remain for fixed-size experiments.
    board.node_target_policy = policy.reshape(action_size // 2, 2)
    board.node_legal_mask = legal.reshape(action_size // 2, 2)
    board.sample_weight = torch.tensor([weight], dtype=torch.float32)
    board.depth = torch.tensor([depth], dtype=torch.long)
    return board


def build_exact_dataset(game, solver):
    states = list(solver.all_states(terminal=False))
    depth_counts = Counter(
        sum(choice != UNRESOLVED for choice in state) for state in states
    )
    return [exact_example(game, solver, state, depth_counts) for state in states]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    policy_correct = 0
    value_correct = 0
    optimal_mass = 0.0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        logits, values = model(batch)
        legal = batch.legal_mask.bool()
        masked_logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        probabilities = F.softmax(masked_logits, dim=1)
        optimal = batch.target_policy > 0
        choices = masked_logits.argmax(dim=1)
        policy_correct += optimal.gather(1, choices.unsqueeze(1)).sum().item()
        value_correct += (
            (values.view(-1) >= 0) == (batch.target_value.view(-1) > 0)
        ).sum().item()
        optimal_mass += (probabilities * optimal).sum().item()
        total += batch.num_graphs
    return {
        "states": total,
        "policy_correct": int(policy_correct),
        "value_correct": int(value_correct),
        "optimal_mass": optimal_mass / total,
    }


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        logits, values = model(batch)
        legal = batch.legal_mask.bool()
        masked_logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        policy_loss = -(
            batch.target_policy * F.log_softmax(masked_logits, dim=1)
        ).sum(dim=1)
        value_loss = (values.view(-1) - batch.target_value.view(-1)).pow(2)
        weights = batch.sample_weight.view(-1)
        loss = ((policy_loss + value_loss) * weights).sum() / weights.sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / total_graphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--architecture",
        choices=(
            "graph",
            "crossing-mlp",
            "port-graph-transformer",
            "port-transformer-residual",
            "port-transformer-indexed",
            "port-transformer-pd-position",
        ),
        default="graph",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--checkpoint", default="checkpoints/capacity_exact.pth.tar"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive")
    if args.weight_decay < 0 or not 0 <= args.dropout < 1:
        raise SystemExit("weight-decay must be nonnegative and dropout in [0, 1)")
    config.validate()
    if len(config.pd_codes) != 1:
        raise SystemExit("capacity_test.py requires one configured shadow")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    game = KnotGraphGame()
    game.getInitBoard()
    solver = ExactSolver(game.initial_pd_code)
    dataset = build_exact_dataset(game, solver)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    network = NNetWrapper(
        game,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=str(device),
        architecture=args.architecture,
    )
    optimizer = torch.optim.AdamW(
        network.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    network.optimizer = optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate / 100
    )
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Capacity dataset: {len(dataset)} nonterminal states; "
        f"device={device}; seed={args.seed}; "
        f"architecture={args.architecture}; hidden_dim={args.hidden_dim}; "
        f"dropout={args.dropout}; weight_decay={args.weight_decay}"
    )
    best_score = None
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(network.model, train_loader, optimizer, device)
        scheduler.step()
        metrics = evaluate(network.model, eval_loader, device)
        score = (
            min(metrics["policy_correct"], metrics["value_correct"]),
            metrics["policy_correct"] + metrics["value_correct"],
            -loss,
        )
        improved = best_score is None or score > best_score
        if improved:
            best_score = score
            best_metrics = metrics
            network.latest_loss = loss
            network.save_checkpoint(str(checkpoint))

        if improved or epoch == 1 or epoch % 10 == 0:
            total = metrics["states"]
            print(
                f"epoch={epoch:03d} loss={loss:.6f} "
                f"policy={metrics['policy_correct']}/{total} "
                f"value={metrics['value_correct']}/{total} "
                f"optimal_mass={metrics['optimal_mass']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
        if (
            metrics["policy_correct"] == metrics["states"]
            and metrics["value_correct"] == metrics["states"]
        ):
            print(f"CAPACITY_SOLVED: YES at epoch {epoch}")
            break
    else:
        print("CAPACITY_SOLVED: NO")

    print(
        "Best exact counts: "
        f"policy={best_metrics['policy_correct']}/{best_metrics['states']} "
        f"value={best_metrics['value_correct']}/{best_metrics['states']}"
    )
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

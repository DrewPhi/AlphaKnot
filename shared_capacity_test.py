#!/usr/bin/env python3
"""Fit one exact-supervised model to 7_1--7_7 and report each PD code.

This is a shared supervised capacity experiment.  Because every evaluated
shadow supplies training labels, it does not measure held-out generalization.
"""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from capacity_test import build_exact_dataset, train_epoch
from exact_solver import ExactSolver
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from seven_crossing_corpus import corpus_records


@torch.no_grad()
def evaluate_by_shadow(model, loader, device, shadow_names):
    """Return exact policy/value metrics separately for every source PD code."""
    model.eval()
    metrics = {
        name: {
            "states": 0,
            "policy_correct": 0,
            "value_correct": 0,
            "optimal_mass": 0.0,
        }
        for name in shadow_names
    }
    for batch in loader:
        batch = batch.to(device)
        logits, values = model(batch)
        legal = batch.legal_mask.bool()
        masked_logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        probabilities = F.softmax(masked_logits, dim=1)
        optimal = batch.target_policy > 0
        choices = masked_logits.argmax(dim=1)
        policy_correct = optimal.gather(1, choices.unsqueeze(1)).view(-1)
        value_correct = (
            (values.view(-1) >= 0) == (batch.target_value.view(-1) > 0)
        )
        optimal_mass = (probabilities * optimal).sum(dim=1)

        for index, name in enumerate(shadow_names):
            selected = batch.shadow_group.view(-1) == index
            count = int(selected.sum().item())
            if not count:
                continue
            row = metrics[name]
            row["states"] += count
            row["policy_correct"] += int(policy_correct[selected].sum().item())
            row["value_correct"] += int(value_correct[selected].sum().item())
            row["optimal_mass"] += float(optimal_mass[selected].sum().item())
    return metrics


def aggregate_metrics(by_shadow):
    return {
        "states": sum(row["states"] for row in by_shadow.values()),
        "policy_correct": sum(
            row["policy_correct"] for row in by_shadow.values()
        ),
        "value_correct": sum(row["value_correct"] for row in by_shadow.values()),
        "optimal_mass": sum(row["optimal_mass"] for row in by_shadow.values()),
    }


def print_metrics(label, row):
    states = row["states"]
    print(
        f"{label:>9} states={states:5d} "
        f"policy={row['policy_correct']:5d}/{states} "
        f"({row['policy_correct']/states:7.2%}) "
        f"optimal_mass={row['optimal_mass']/states:7.2%} "
        f"value={row['value_correct']:5d}/{states} "
        f"({row['value_correct']/states:7.2%})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument(
        "--checkpoint", default="checkpoints/shared_seven_exact.pth.tar"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any(
        value < 1
        for value in (
            args.epochs,
            args.batch_size,
            args.report_every,
            args.hidden_dim,
            args.num_heads,
            args.num_layers,
        )
    ):
        raise SystemExit("size, depth, epochs, batch-size, and report-every must be positive")
    if args.hidden_dim % args.num_heads:
        raise SystemExit("hidden-dim must be divisible by num-heads")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = corpus_records()
    names = [name for name, _ in records]
    datasets = []
    games = []
    for shadow_group, (name, pd_code) in enumerate(records):
        game = KnotGraphGame(pd_code=pd_code)
        game.getInitBoard()
        solver = ExactSolver(pd_code)
        dataset = build_exact_dataset(game, solver)
        for graph in dataset:
            # Avoid names containing "index": PyG treats those as connectivity
            # tensors and offsets them while collating separate graphs.
            graph.shadow_group = torch.tensor([shadow_group], dtype=torch.long)
        games.append(game)
        datasets.extend(dataset)
        print(f"Exact table {name}: {len(dataset)} nonterminal states")

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        datasets,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    eval_loader = DataLoader(datasets, batch_size=args.batch_size, shuffle=False)
    network = NNetWrapper(
        games[0],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=str(device),
        architecture="port-transformer-pd-position",
    )
    optimizer = torch.optim.AdamW(
        network.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate / 100
    )
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in network.model.parameters())

    print(
        f"Shared dataset: {len(datasets)} states across {len(names)} PD codes; "
        f"device={device}; hidden_dim={args.hidden_dim}; "
        f"heads={args.num_heads}; layers={args.num_layers}; "
        f"parameters={parameter_count:,}; seed={args.seed}"
    )
    best_score = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(network.model, train_loader, optimizer, device)
        scheduler.step()
        by_shadow = evaluate_by_shadow(network.model, eval_loader, device, names)
        total = aggregate_metrics(by_shadow)
        minimum_shadow_accuracy = min(
            min(
                row["policy_correct"] / row["states"],
                row["value_correct"] / row["states"],
            )
            for row in by_shadow.values()
        )
        score = (
            minimum_shadow_accuracy,
            min(total["policy_correct"], total["value_correct"]),
            total["policy_correct"] + total["value_correct"],
            -loss,
        )
        improved = best_score is None or score > best_score
        if improved:
            best_score = score
            network.latest_loss = loss
            network.save_checkpoint(str(checkpoint))

        if improved or epoch == 1 or epoch % args.report_every == 0:
            print(f"epoch={epoch:03d} loss={loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}")
            print_metrics("combined", total)

        if (
            total["policy_correct"] == total["states"]
            and total["value_correct"] == total["states"]
        ):
            print(f"SHARED_CAPACITY_SOLVED: YES at epoch {epoch}")
            break
    else:
        print("SHARED_CAPACITY_SOLVED: NO")

    network.load_checkpoint(str(checkpoint), load_optimizer=False)
    final_by_shadow = evaluate_by_shadow(network.model, eval_loader, device, names)
    print("=== Best-checkpoint exact results by source PD code ===")
    for name in names:
        print_metrics(name, final_by_shadow[name])
    final_total = aggregate_metrics(final_by_shadow)
    print_metrics("combined", final_total)
    for name, row in final_by_shadow.items():
        solved = (
            row["policy_correct"] == row["states"]
            and row["value_correct"] == row["states"]
        )
        print(f"{name} SOLVED: {'YES' if solved else 'NO'}")
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

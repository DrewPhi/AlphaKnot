#!/usr/bin/env python3
"""Fit one exact-supervised model to standard prime diagrams of mixed sizes.

This is a representational-capacity experiment.  Every evaluated exact state
also supplies a training label; it is not a self-play or generalization result.
"""

import argparse
from collections import Counter
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from capacity_test import build_exact_dataset
from exact_solver import ExactSolver
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from prime_knot_corpus import corpus_records


def dense_action_targets(batch):
    """Pad nodewise keep/switch targets to the model's batch action width."""
    target, _ = to_dense_batch(batch.node_target_policy, batch.batch)
    legal, _ = to_dense_batch(batch.node_legal_mask, batch.batch)
    return target.flatten(1), legal.flatten(1).bool()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        logits, values = model(batch)
        target, legal = dense_action_targets(batch)
        masked_logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        policy_loss = -(target * F.log_softmax(masked_logits, dim=1)).sum(dim=1)
        value_loss = (values.view(-1) - batch.target_value.view(-1)).pow(2)
        loss = (policy_loss + value_loss).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / total_graphs


@torch.no_grad()
def evaluate(model, loader, device, names):
    model.eval()
    metrics = {
        name: {"states": 0, "policy_correct": 0, "value_correct": 0, "optimal_mass": 0.0}
        for name in names
    }
    for batch in loader:
        batch = batch.to(device)
        logits, values = model(batch)
        target, legal = dense_action_targets(batch)
        masked_logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        probabilities = F.softmax(masked_logits, dim=1)
        optimal = target > 0
        choices = masked_logits.argmax(dim=1)
        policy_correct = optimal.gather(1, choices.unsqueeze(1)).view(-1)
        value_correct = (values.view(-1) >= 0) == (batch.target_value.view(-1) > 0)
        optimal_mass = (probabilities * optimal).sum(dim=1)

        for group, name in enumerate(names):
            selected = batch.shadow_group.view(-1) == group
            count = int(selected.sum().item())
            if not count:
                continue
            row = metrics[name]
            row["states"] += count
            row["policy_correct"] += int(policy_correct[selected].sum().item())
            row["value_correct"] += int(value_correct[selected].sum().item())
            row["optimal_mass"] += float(optimal_mass[selected].sum().item())
    return metrics


def aggregate(rows):
    rows = tuple(rows)
    return {
        key: sum(row[key] for row in rows)
        for key in ("states", "policy_correct", "value_correct", "optimal_mass")
    }


def print_metrics(label, row):
    states = row["states"]
    print(
        f"{label:>9} states={states:6d} "
        f"policy={row['policy_correct']:6d}/{states} ({row['policy_correct']/states:7.2%}) "
        f"optimal_mass={row['optimal_mass']/states:7.2%} "
        f"value={row['value_correct']:6d}/{states} ({row['value_correct']/states:7.2%})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-crossings", type=int, default=3)
    parser.add_argument("--max-crossings", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/prime_3_to_8_variable.pth.tar")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not 3 <= args.min_crossings <= args.max_crossings <= 8:
        raise SystemExit("current versioned corpus supports crossing counts 3 through 8")
    if args.hidden_dim % args.num_heads:
        raise SystemExit("hidden-dim must be divisible by num-heads")
    if min(args.epochs, args.eval_every, args.batch_size, args.num_layers) < 1:
        raise SystemExit("epochs, eval-every, batch-size, and layers must be positive")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise SystemExit("warmup-epochs must be nonnegative and less than epochs")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = corpus_records(args.min_crossings, args.max_crossings)
    names = [name for name, _ in records]
    knot_counts = Counter(len(pd_code) for _, pd_code in records)
    games = []
    dataset = []
    sampling_weights = []
    for group, (name, pd_code) in enumerate(records):
        game = KnotGraphGame(pd_code=pd_code)
        game.getInitBoard()
        examples = build_exact_dataset(game, ExactSolver(pd_code))
        depth_levels = len(pd_code) + 1
        for graph in examples:
            # Fixed-size experiments use graph-level [1, 2n] tensors, which
            # cannot be concatenated when n differs.  The node-level copies
            # carry the same supervision and pad only after batching.
            del graph.target_policy
            del graph.legal_mask
            graph.shadow_group = torch.tensor([group], dtype=torch.long)
            # Hierarchical sampling: equal total mass per crossing number,
            # knot within that size, and game depth within that knot.
            sampling_weights.append(
                float(graph.sample_weight.item())
                / (knot_counts[len(pd_code)] * depth_levels)
            )
        games.append(game)
        dataset.extend(examples)
        print(f"Exact table {name}: {len(examples)} nonterminal states")

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sampling_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    network = NNetWrapper(
        games[0],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=str(device),
        architecture="variable-port-transformer",
    )
    optimizer = torch.optim.AdamW(
        network.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    network.optimizer = optimizer
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.learning_rate / 100,
    )
    if args.warmup_epochs:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0 / args.warmup_epochs,
                    total_iters=args.warmup_epochs,
                ),
                cosine,
            ],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = cosine

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in network.model.parameters())
    print(
        f"Mixed-size dataset: {len(dataset)} states, {len(names)} knots, "
        f"crossings={args.min_crossings}..{args.max_crossings}; device={device}; "
        f"hidden={args.hidden_dim}; heads={args.num_heads}; layers={args.num_layers}; "
        f"parameters={parameter_count:,}; seed={args.seed}"
    )

    best_score = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(network.model, train_loader, optimizer, device)
        scheduler.step()
        if epoch != 1 and epoch % args.eval_every and epoch != args.epochs:
            continue
        by_knot = evaluate(network.model, eval_loader, device, names)
        total = aggregate(by_knot.values())
        minimum_accuracy = min(
            min(row["policy_correct"], row["value_correct"]) / row["states"]
            for row in by_knot.values()
        )
        score = (
            minimum_accuracy,
            min(total["policy_correct"], total["value_correct"]),
            total["policy_correct"] + total["value_correct"],
            -loss,
        )
        if best_score is None or score > best_score:
            best_score = score
            network.latest_loss = loss
            network.save_checkpoint(str(checkpoint))
        print(f"epoch={epoch:03d} loss={loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}")
        print_metrics("combined", total)
        for crossings in range(args.min_crossings, args.max_crossings + 1):
            print_metrics(
                f"n={crossings}",
                aggregate(row for name, row in by_knot.items() if name.startswith(f"{crossings}_")),
            )
        if all(
            row["policy_correct"] == row["states"]
            and row["value_correct"] == row["states"]
            for row in by_knot.values()
        ):
            print(f"MIXED_SIZE_CAPACITY_SOLVED: YES at epoch {epoch}")
            break
    else:
        print("MIXED_SIZE_CAPACITY_SOLVED: NO")

    network.load_checkpoint(str(checkpoint), load_optimizer=False)
    final = evaluate(network.model, eval_loader, device, names)
    print("=== Best-checkpoint exact results by knot ===")
    for name in names:
        print_metrics(name, final[name])
        solved = (
            final[name]["policy_correct"] == final[name]["states"]
            and final[name]["value_correct"] == final[name]["states"]
        )
        print(f"{name} SOLVED: {'YES' if solved else 'NO'}")
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

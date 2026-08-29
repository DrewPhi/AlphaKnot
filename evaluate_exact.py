#!/usr/bin/env python3
"""Compare a trained AlphaKnot checkpoint with the complete minimax table."""

import argparse
from collections import defaultdict
import json

import numpy as np

import config
from exact_solver import ExactSolver, UNRESOLVED
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper


def board_from_state(game, state):
    board = game.pd_code_to_graph_data([list(c) for c in game.initial_pd_code])
    player = 1
    for crossing, choice in enumerate(state):
        if choice != UNRESOLVED:
            board, player = game.getNextState(board, player, 2 * crossing + choice)
    return board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best.pth.tar")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--architecture",
        choices=(
            "graph",
            "crossing-mlp",
            "port-graph-transformer",
            "port-transformer-residual",
            "port-transformer-indexed",
            "port-transformer-pd-position",
            "pd-state-mlp",
        ),
        default="graph",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--pd-code-json",
        help="optional one-based oriented PD code as a JSON list of 4-tuples",
    )
    args = parser.parse_args()

    config.validate()
    pd_code = None
    if args.pd_code_json:
        try:
            pd_code = json.loads(args.pd_code_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --pd-code-json: {exc}") from exc
        if not isinstance(pd_code, list) or not pd_code or any(
            not isinstance(crossing, list) or len(crossing) != 4
            for crossing in pd_code
        ):
            raise SystemExit("--pd-code-json must be a nonempty JSON list of 4-lists")
    game = KnotGraphGame(pd_code=pd_code)
    game.getInitBoard()
    solver = ExactSolver(game.initial_pd_code)
    network = NNetWrapper(
        game,
        hidden_dim=args.hidden_dim,
        device=args.device,
        architecture=args.architecture,
    )
    network.load_checkpoint(args.checkpoint, load_optimizer=False)

    by_depth = defaultdict(lambda: {"states": 0, "top1": 0, "mass": 0.0, "value": 0})
    for state in solver.all_states(terminal=False):
        depth = sum(choice != UNRESOLVED for choice in state)
        board = board_from_state(game, state)
        policy, predicted_value = network.predict(board)
        legal = np.zeros_like(policy)
        legal[list(solver.legal_actions(state))] = 1
        policy = policy * legal
        policy /= policy.sum()
        optimal = solver.optimal_actions(state)
        exact_value = solver.value(state)
        row = by_depth[depth]
        row["states"] += 1
        row["top1"] += int(int(np.argmax(policy)) in optimal)
        row["mass"] += float(policy[list(optimal)].sum())
        row["value"] += int((predicted_value >= 0) == (exact_value > 0))

    print("depth states top1-optimal optimal-policy-mass value-sign")
    total = {"states": 0, "top1": 0, "mass": 0.0, "value": 0}
    for depth in sorted(by_depth):
        row = by_depth[depth]
        count = row["states"]
        print(
            f"{depth:5d} {count:6d} {row['top1']/count:12.2%} "
            f"{row['mass']/count:19.2%} {row['value']/count:10.2%}"
        )
        for key in total:
            total[key] += row[key]
    count = total["states"]
    print(
        f"total {count:6d} {total['top1']/count:12.2%} "
        f"{total['mass']/count:19.2%} {total['value']/count:10.2%}"
    )
    root = solver.initial_state
    print(f"Exact root value: {solver.value(root):+d}")
    print(f"Exact optimal root actions: {solver.optimal_actions(root)}")
    policy_solved = total["top1"] == count
    value_solved = total["value"] == count
    print(f"Policy optimal on every nonterminal state: {'YES' if policy_solved else 'NO'}")
    print(f"Value sign correct on every nonterminal state: {'YES' if value_solved else 'NO'}")
    print(f"SOLVED: {'YES' if policy_solved and value_solved else 'NO'}")


if __name__ == "__main__":
    main()

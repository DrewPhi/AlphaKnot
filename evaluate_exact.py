#!/usr/bin/env python3
"""Compare a trained AlphaKnot checkpoint with the complete minimax table."""

import argparse
from collections import defaultdict

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
    args = parser.parse_args()

    config.validate()
    game = KnotGraphGame()
    game.getInitBoard()
    solver = ExactSolver(game.initial_pd_code)
    network = NNetWrapper(game, device=args.device)
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

#!/usr/bin/env python3
"""Exact minimax solution of the configured crossing-resolution game."""

import argparse
from functools import lru_cache
from itertools import product

import config
from knot_invariants import jones_is_one
from pd_code_utils import flip_crossing


UNRESOLVED = -1


class ExactSolver:
    """Solve one fixed shadow; values are from the player-to-move viewpoint."""

    def __init__(self, pd_code=None, knotter_first=None):
        source = pd_code if pd_code is not None else config.pd_codes[0]
        self.pd_code = tuple(tuple(map(int, crossing)) for crossing in source)
        self.crossings = len(self.pd_code)
        self.n_edges = max(label for crossing in self.pd_code for label in crossing)
        self.knotter_first = (
            config.knotter_first if knotter_first is None else knotter_first
        )
        self.initial_state = (UNRESOLVED,) * self.crossings

    @staticmethod
    def player_to_move(state):
        return 1 if sum(choice != UNRESOLVED for choice in state) % 2 == 0 else -1

    def resolved_pd_code(self, state):
        if UNRESOLVED in state:
            raise ValueError("state is not terminal")
        return [
            list(crossing) if choice == 0
            else flip_crossing(crossing, self.n_edges)
            for crossing, choice in zip(self.pd_code, state)
        ]

    @lru_cache(maxsize=None)
    def is_unknot(self, state):
        return jones_is_one(self.resolved_pd_code(state))

    def winner(self, state):
        """Return the winning player identity (+1 or -1) at a terminal state."""
        unknotter = -1 if self.knotter_first else 1
        return unknotter if self.is_unknot(state) else -unknotter

    @lru_cache(maxsize=None)
    def value(self, state):
        """Return exact value (+1/-1) for the player whose turn it is."""
        if UNRESOLVED not in state:
            return 1 if self.winner(state) == self.player_to_move(state) else -1
        return max(-self.value(self.play(state, action)) for action in self.legal_actions(state))

    def play(self, state, action):
        crossing, choice = divmod(action, 2)
        if crossing >= self.crossings or state[crossing] != UNRESOLVED:
            raise ValueError(f"illegal action {action} for {state}")
        child = list(state)
        child[crossing] = choice
        return tuple(child)

    def legal_actions(self, state):
        return tuple(
            2 * crossing + choice
            for crossing, current in enumerate(state)
            if current == UNRESOLVED
            for choice in (0, 1)
        )

    def optimal_actions(self, state):
        target = self.value(state)
        return tuple(
            action for action in self.legal_actions(state)
            if -self.value(self.play(state, action)) == target
        )

    def all_states(self, terminal=None):
        for state in product((UNRESOLVED, 0, 1), repeat=self.crossings):
            is_terminal = UNRESOLVED not in state
            if terminal is None or terminal == is_terminal:
                yield state

    def principal_variation(self):
        state = self.initial_state
        actions = []
        while UNRESOLVED in state:
            action = min(self.optimal_actions(state))
            actions.append(action)
            state = self.play(state, action)
        return actions, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config.validate()
    if len(config.pd_codes) != 1:
        raise SystemExit("exact_solver.py currently requires exactly one configured shadow")

    solver = ExactSolver()
    root_value = solver.value(solver.initial_state)
    root_winner = solver.player_to_move(solver.initial_state) if root_value == 1 else -1
    actions, terminal = solver.principal_variation()
    terminals = list(solver.all_states(terminal=True))
    unknots = sum(solver.is_unknot(state) for state in terminals)

    role = "knotter" if root_winner == (1 if config.knotter_first else -1) else "unknotter"
    print(f"Exact states: {3 ** solver.crossings} ({len(terminals)} terminal)")
    print(f"Terminal Jones=1: {unknots}; Jones!=1: {len(terminals) - unknots}")
    print(f"Forced winner: player {root_winner:+d} ({role})")
    print(f"Optimal root actions: {solver.optimal_actions(solver.initial_state)}")
    print(f"One principal variation: {actions}")
    print(f"Principal-variation terminal is unknot: {solver.is_unknot(terminal)}")


if __name__ == "__main__":
    main()

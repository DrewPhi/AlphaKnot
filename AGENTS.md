# AlphaKnot Agent Guide

This file applies to the entire repository. It is the operational contract for
human contributors and coding agents.

## Current milestone

Do not broaden the production game to larger or mixed knot shadows until the
configured seven-crossing game is solved by the learned agent under the exact
evaluation in `evaluate_exact.py`.

For this milestone, `SOLVED: YES` means both of the following hold on every one
of the 2,059 nonterminal states:

1. The network's highest-probability legal action is minimax-optimal.
2. The sign of the network value agrees with the exact minimax value.

Arena wins or wins against random players are useful diagnostics, but they are
not evidence that the game is solved. The exact table is authoritative.

## Mathematical invariants

- Production inputs are oriented, one-component PD codes.
- Edge labels are consecutive and one-based.
- In `[a, b, c, d]`, `(a, c)` is the under-strand and `(b, d)` is the
  over-strand.
- Labels advance cyclically, including the final-label-to-1 transition.
- A crossing change must use `pd_code_utils.flip_crossing`; do not invent a
  second sign convention.
- The configured terminal classifier is the normalized Jones-polynomial
  criterion in `knot_invariants.py`. Treat `Jones == 1` as the game rule, not as
  a general theorem characterizing the unknot.
- Do not raise `max_validated_crossings` without independent validation data and
  tests.

## Sources of truth

Use evidence in this order:

1. `exact_solver.py` for minimax values and optimal actions.
2. `evaluate_exact.py` for learned-policy and learned-value agreement.
3. Adversarial arena evaluation.
4. Random-opponent win rates.

Never describe a model as perfect, solved, or near-perfect using only arena or
random-play results.

`capacity_test.py` uses exact labels for training and evaluation. Report its
result only as representational capacity or memorization; do not describe it as
self-play discovery or generalization.

## Required checks

Run from the repository root:

```bash
make test
make exact
make smoke
```

At minimum, every change to game logic, PD conversion, terminal evaluation,
MCTS, or exact evaluation must pass `make test` and `make exact`. Changes to
multiprocessing, checkpoints, the network, training, or arena code must also
pass `make smoke`.

Keep regression tests small and deterministic. Every mathematical bug fix needs
a test that fails under the old behavior.

## Experiment discipline

- Record meaningful completed runs in `docs/EXPERIMENTS.md`.
- Include the Git commit, scheduler job ID, configuration, elapsed time, exact
  metrics, and final verdict.
- Keep checkpoints, training histories, scheduler output, and other generated
  artifacts out of Git.
- Do not compare experiments whose PD code, player roles, exact evaluator, or
  solved criterion changed without saying so explicitly.
- Preserve seeds and environment overrides when reproducibility matters.

## Bouchet workflow

- Install or compile dependencies on an allocated compute node, not a login
  node.
- Use the `alphaknot` Conda environment and the current Miniconda module named in
  the job script.
- The environment's `$CONDA_PREFIX/lib` must lead dynamic-library resolution so
  SnapPy/SQLite use the matching Conda C++ runtime.
- Use `gpu_devel` for bounded validation and `gpu` only after the pipeline and
  resource request are validated.
- Inspect `sacct` and the complete scheduler log before reporting success.
- Never commit cluster credentials, SSH sockets, checkpoints, or logs.

## Repository boundaries

- `main` contains the production implementation only.
- Historical experiments remain on `archive/legacy-2026-08-20`.
- Preserve unrelated user changes in a dirty worktree.
- Stage explicit paths; do not use broad staging commands.
- Work on a feature branch and merge only after the required checks pass.

## File map

- `config.py`: production shadow and experiment settings.
- `knot_graph_game.py`: game rules and graph state transitions.
- `pd_code_utils.py`: PD conventions and graph/PD conversion.
- `knot_invariants.py`: exhaustive small-knot Jones evaluator.
- `exact_solver.py`: complete minimax ground truth.
- `evaluate_exact.py`: exhaustive learned-model evaluation.
- `capacity_test.py`: exact-table architecture capacity experiment.
- `mcts.py`, `coach.py`, `arena.py`: search, training, and match evaluation.
- `knot_graph_nnet.py`: graph policy/value network.
- `jobscript_bouchet_full.sh`: end-to-end Bouchet validation and training.

## Before merging

- Confirm the worktree contains no generated artifacts.
- Run the checks appropriate to the changed files.
- Update documentation when commands, conventions, or solved criteria change.
- Add a row to the experiment ledger when reporting a new scientific result.
- State limitations plainly; exploratory embeddings and larger unsolved games
  must not be presented as certified results.

# AlphaKnot Agent Guide

This file applies to the entire repository. It is the operational contract for
human contributors and coding agents.

## Current milestone

The fixed seven-crossing architecture-capacity gate is complete. The tuned
PD-position port transformer reached exhaustive `SOLVED: YES` on all 2,059
nonterminal states in Bouchet job `23989965`. This was supervised exact-table
training: it proves representational capacity, not AlphaZero self-play discovery
or generalization.

The current milestone is a single shared model over a versioned corpus of
seven-crossing, oriented, one-component PD shadows. Complete the seven-crossing
dataset, variable-corpus training path, and held-out evaluation before adding
eight-crossing shadows to production training. Follow `docs/ROADMAP.md` and
`docs/STRATEGY_EMBEDDINGS.md`.

For every shadow with an exact table, exact `SOLVED: YES` still means both:

1. The highest-probability legal action is minimax-optimal at every nonterminal
   state.
2. The value sign agrees with exact minimax at every nonterminal state.

Arena wins or wins against random players are useful diagnostics, but they are
not evidence that the game is solved. The exact table is authoritative.

Do not call a shared model generalized merely because it fits a mixed training
corpus. Generalization claims require frozen shadow- or knot-type-disjoint test
splits and must report whether exact labels were used during training.

## Mathematical invariants

- Production inputs are oriented, one-component PD codes.
- Edge labels are consecutive and one-based.
- In `[a, b, c, d]`, `(a, c)` is the under-strand and `(b, d)` is the
  over-strand.
- Labels advance cyclically, including the final-label-to-1 transition.
- Canonicalize external PD codes with `pd_code_utils.canonicalize_pd_code`;
  equivalent basepoints, component orientations, and crossing-list orders must
  produce identical model inputs.
- Preserve the returned source/canonical action maps at API boundaries.
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
- Version corpus manifests and frozen train/validation/test splits. Canonical
  relabelings or alternate diagrams of one object must not leak across a split.
- Distinguish the object being studied: a PD shadow game, a particular diagram,
  or a knot type. Strategy belongs directly to the game on a shadow.
- Use the canonical model-independent probe panels in
  `docs/STRATEGY_EMBEDDINGS.md` when comparing activation diffusion operators.

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
- `docs/ROADMAP.md`: staged generalization and paper milestones.
- `docs/STRATEGY_EMBEDDINGS.md`: activation and MCTS diffusion protocol.

## Before merging

- Confirm the worktree contains no generated artifacts.
- Run the checks appropriate to the changed files.
- Update documentation when commands, conventions, or solved criteria change.
- Add a row to the experiment ledger when reporting a new scientific result.
- State limitations plainly; exploratory embeddings and larger unsolved games
  must not be presented as certified results.

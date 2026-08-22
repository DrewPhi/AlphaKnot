# AlphaKnot

AlphaKnot is an AlphaZero-style self-play system for a two-player knot game.
States are oriented planar-diagram (PD) codes, search uses MCTS, and the
policy/value model is a PyTorch Geometric graph transformer.

This branch contains only the production GPU-training implementation. Historical
experiments, checkpoints, and training examples are retained on the archival Git
branch and intentionally excluded from `main`.

## Supported mathematical convention

The game currently accepts one-component PD codes with consecutive, one-based
edge labels. For a crossing `[a, b, c, d]`, `(a, c)` is the under-strand and
`(b, d)` is the over-strand. Labels advance cyclically, including `n -> 1`.

Terminal classification via Alexander and Jones polynomials is limited by
`max_validated_crossings` in `config.py`. Extending that bound requires an
independent validation dataset; these polynomials do not detect the unknot in
general by a theorem used here.

## Environment

Create an isolated Python 3.11/3.12 environment and install a PyTorch build that
matches the cluster CUDA driver. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

PyTorch Geometric extension wheels such as `torch-scatter` must match the exact
PyTorch/CUDA versions. Follow the PyTorch Geometric installation matrix rather
than reusing an extension compiled for another PyTorch release.

For tests:

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python tests/smoke_multiprocessing.py
python tests/smoke_arena.py
```

## Training

Start a new run with:

```bash
python main.py
```

Resume from `checkpoints/best.pth.tar` with:

```bash
ALPHAKNOT_RESUME=1 python main.py
```

For the two-GPU SLURM configuration:

```bash
sbatch jobscript.sh
```

On Yale Bouchet, first run the exact 7-crossing minimax baseline and then
submit the short `gpu_devel` validation job:

```bash
python exact_solver.py
sbatch jobscript_bouchet_smoke.sh
```

The exact solver evaluates all 2,187 partial crossing assignments. After
training, `evaluate_exact.py` compares the checkpoint's policy and value at
every nonterminal state with the minimax table. The smoke job is deliberately
small; it validates the pipeline and should not be treated as a converged run.

For a single Bouchet job that installs test-only dependencies, runs all tests,
trains, and prints an explicit exhaustive `SOLVED: YES/NO` verdict, use:

```bash
sbatch jobscript_bouchet_full.sh
```

The optional SnapPy validation dependency is installed through the same Conda
environment from PyPI; SageMath is not required for the production evaluator.

`torchrun` uses both GPUs for DDP training. Rank 0 uses CPU processes for
self-play and arena games, bounded by `SLURM_CPUS_PER_TASK`. This avoids placing
one model replica on a GPU for every CPU worker. Tune `selfplay_workers`, memory,
and MCTS counts only after profiling the target node.

Checkpoints and serialized examples are runtime artifacts and are ignored by
Git. Store production artifacts in cluster/object storage rather than committing
them to repository history.

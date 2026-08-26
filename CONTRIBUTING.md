# Contributing to AlphaKnot

Read `AGENTS.md` before making changes. It contains the mathematical
conventions, exact-solution gate, validation requirements, and cluster rules.

## Development setup

Create an isolated Python 3.11 or 3.12 environment, install a PyTorch build
appropriate for the machine, and then install the project dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

## Validation

Use the repository commands rather than maintaining private variants:

```bash
make test
make exact
make smoke
```

The exact evaluator is the acceptance test for claims about optimal play. A
random-opponent or arena win rate cannot replace it.

## Change workflow

1. Start from an up-to-date `main` and create a focused feature branch.
2. Add or update regression tests with the implementation.
3. Run the required checks from `AGENTS.md`.
4. Keep generated models, training examples, and Slurm logs outside Git.
5. Document material experiments in `docs/EXPERIMENTS.md`.
6. Use a concise commit message that describes the behavioral change.

Do not mix mathematical convention changes with unrelated model or scheduler
changes. Convention changes need explicit examples and independent validation.

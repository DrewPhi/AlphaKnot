# AlphaKnot Research Roadmap

This document tracks planned work and acceptance gates. Completed numerical
runs belong in `EXPERIMENTS.md`.

## Scientific objective

Train a shared AlphaZero-style agent on games defined by oriented PD shadows,
then represent each learned strategy through diffusion geometry constructed
from internal activations and search behavior. Test whether strategic geometry
is stable across training seeds and related to classical knot invariants.

Be precise about the object of study. An unresolved PD shadow defines the game.
A knot type may have multiple diagrams, and multiple diagrams may induce the
same or related shadows. A knot-level claim requires robustness across those
diagram choices.

## Completed foundation

- Exact enumeration and minimax evaluation for the original seven-crossing
  shadow: 2,187 total states and 2,059 nonterminal states.
- Solved width-256 crossing-state MLP capacity control.
- PD-native four-port graph representation with typed cyclic, opposite-strand,
  and same-arc relations.
- Deterministic first-encounter PD traversal positional encoding.
- Versioned oriented-PD canonicalization with reversible action maps for
  traversal basepoint, component orientation, and crossing-list order.
- Solved PD-position port-transformer capacity run: job `23989965`, epoch 512,
  100% exact policy and value-sign agreement.

These are supervised capacity results. The original AlphaZero self-play run did
not solve the fixed shadow.

## Milestone G1: shared seven-crossing model

### G1.1 Corpus and manifest

Create a versioned manifest of seven-crossing, oriented, one-component PD
shadows. Each record must contain:

- stable `shadow_id` and source identifier;
- source knot/table name when applicable;
- canonical ordered PD code and crossing count;
- canonicalization version;
- equivalence group for crossing-list permutations, valid arc relabelings,
  alternate diagrams, and duplicate shadows;
- train, validation, or test split;
- exact-table availability and checksum.

Start with shadows derived from the tabulated seven-crossing prime knots, but
deduplicate at the shadow level and retain the source-diagram relationship. Do
not use the phrase "all seven-crossing knots" until the manifest states exactly
which knot table and diagram-generation convention make the corpus complete.

### G1.2 Generalized training path

- Replace single-shadow configuration assumptions with manifest-driven loading.
- Support batches containing different PD shadows with a common crossing count.
- Derive positional identity independently for every PD code.
- Preserve crossing-list permutation equivariance and valid relabeling tests.
- Produce masked per-crossing `keep/switch` actions from one shared checkpoint.
- Keep the fixed-shadow MLP and exact-table transformer as controls.

### G1.3 Frozen evaluations

Report three distinct settings:

1. **Per-shadow capacity:** exact-table supervision and exhaustive evaluation.
2. **Shared supervised capacity:** one model fit to exact tables from multiple
   training shadows.
3. **Generalization:** evaluate frozen held-out shadows or knot types without
   training on their exact tables.

For every exact-evaluable shadow, report top-policy optimality, optimal-policy
mass, value-sign accuracy, and `SOLVED: YES/NO`. Aggregate metrics must not hide
individual failed shadows.

### G1.4 Self-play

After the shared architecture passes supervised capacity checks, train it using
AlphaZero self-play. Compare raw network policy with fixed-budget MCTS policy and
exact minimax on every tractable shadow. A self-play solution must not load exact
policy or value targets.

### G1 exit gate

- Corpus manifest and splits are frozen and reproducible.
- One shared checkpoint runs every seven-crossing corpus member.
- Exact metrics are reported per shadow on train, validation, and test splits.
- At least three seeds establish optimization variance.
- Relabeling and crossing-list permutation tests pass.
- Capacity, self-play, and held-out generalization claims are clearly separated.

## Milestone G2: eight crossings

Extend the same manifest and model path to eight-crossing shadows only after G1.
An eight-crossing shadow has 6,561 ternary partial assignments and 6,305
nonterminal states, so exhaustive exact tables remain a plausible evaluation
tool before scaling further.

G2 must add mixed-size batching or padding/action masks if seven- and
eight-crossing shadows are trained together. Report performance separately by
crossing count and retain knot-type-disjoint evaluation.

## Milestone R1: strategic diffusion geometry

Implement the frozen probe panels and activation/MCTS exporters described in
`STRATEGY_EMBEDDINGS.md`. Begin within one crossing count, where canonical
ternary resolution patterns align datapoints exactly across shadows and models.

Required controls include untrained networks, topology-only features, raw
network policy, MCTS policy, multiple seeds, and multiple diagrams for the same
knot type. Visualizations such as PHATE are exploratory; quantitative neighbor,
prediction, and stability tests are required for paper claims.

## Near-term implementation order

1. Define and validate the seven-crossing corpus manifest.
2. Refactor configuration and datasets for multiple shadows.
3. Train the shared model with exact supervision as a capacity control.
4. Add frozen held-out evaluation and per-shadow reporting.
5. Train the shared AlphaZero self-play model.
6. Export canonical activations and MCTS statistics.
7. Construct and compare diffusion operators.
8. Extend the validated pipeline to eight crossings.

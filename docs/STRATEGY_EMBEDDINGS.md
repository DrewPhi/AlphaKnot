# Strategy Activation and Diffusion Protocol

This document defines the planned representation-analysis protocol. It is a
design specification, not a record of completed results.

## Primary question

Does the diffusion geometry of neural activations encode learned knot-game
strategy, remain stable across optimization seeds, and organize PD shadows in a
way related to classical knot invariants?

## Models

The primary analysis uses one shared generalized PD-position port transformer.
A shared model places activations from every shadow in one coordinate system and
tests transfer directly.

Separately trained per-shadow experts remain valuable controls. When comparing
experts, use multiple seeds to estimate within-shadow training variation before
interpreting between-shadow operator distances as mathematical structure.

## Canonical datapoints

A datapoint is a nonterminal partial crossing-resolution state, not a selected
move or a whole knot. For an `n`-crossing shadow, use the canonical PD traversal
order and a ternary vector

```text
(r_1, ..., r_n),  r_i in {unresolved, original, switched}.
```

For shadows with the same crossing count, apply the same ordered ternary pattern
to each shadow. The probe panel must be fixed before model evaluation and must
not depend on which states a particular agent happens to visit.

- Seven crossings: use all 2,059 nonterminal patterns.
- Eight crossings: use all 6,305 nonterminal patterns when feasible.
- Larger boards: use a frozen, depth-balanced, model-independent panel generated
  with common random numbers or a low-discrepancy sequence.

Keep terminal states in a separate outcome analysis. Do not mix them into the
primary strategy operator.

## Activation locations

Export activations at several named stages and build a separate operator for
each:

1. port tokens after local typed message passing;
2. canonically ordered crossing tokens after global attention;
3. final graph token;
4. penultimate policy-head representation or masked logits;
5. penultimate value-head representation.

The final crossing-token operator is the primary candidate because policy
actions are produced per crossing. The graph-token operator emphasizes global
evaluation. Layerwise comparison tests where strategic and invariant-related
geometry first emerges.

Record model commit, checkpoint hash, training seed, layer name, canonicalizer
version, probe-panel checksum, dtype, and extraction settings with every matrix.

## Activation diffusion operator

For model or shadow `K`, let `X_K` contain one activation row per canonical probe
state. Compute the established research pipeline:

1. pairwise activation distances;
2. an explicitly versioned kernel and bandwidth rule;
3. affinity matrix `W_K`;
4. row-normalized diffusion operator
   `P_K = diag(W_K 1)^{-1} W_K`.

Use identical probe ordering and kernel settings when directly comparing
operators. Construct depth-specific operators as well as a combined operator
whose weighting does not let the numerous late-depth states dominate.

## MCTS behavioral operator

For every canonical probe state, run a fixed search budget and save the complete
legal-action vectors:

- raw network prior `p(a|s)`;
- MCTS visit distribution `pi(a|s)`;
- visit counts `N(s,a)`;
- backed-up action values `Q(s,a)`;
- network value `v(s)`.

Do not use only the top moves for the primary representation. Top-k summaries
discard symmetry, uncertainty, and critical low-probability alternatives. They
may be used later for interpretation.

Construct a behavioral distance/operator from the full masked vectors. Compare
it with activation operators to test whether internal geometry reflects raw or
search-improved strategy. Track search correction, for example
`KL(pi_MCTS || p_network)`, using a fixed MCTS budget and temperature.

## Depth and sampling

Report operators and comparisons by number of resolved crossings. Also report a
depth-balanced aggregate. On-policy occupancy is a secondary analysis because
model-specific trajectories change the datapoint distribution; every model
must first be evaluated on the common probe panel.

### Cross-crossing matched pilot panel

The first 3--8 crossing PHATE pilot uses the versioned panel
`repeat-all-n3-nonterminal-v1`. Its 19 probe IDs are all nonterminal ternary
states on three crossings. For an `n`-crossing diagram, each three-coordinate
pattern repeats periodically through canonical PD traversal order. This gives
every knot the same 19 probe identities, a 19-by-hidden-width activation matrix,
and a directly comparable 19-by-19 diffusion operator.

This small cross-size panel is an exploratory alignment device, not a
replacement for the complete 2,059-state seven-crossing and 6,305-state
eight-crossing operators. Report the full within-crossing operators separately;
test alternative common-random-number panels before treating cross-size PHATE
neighborhoods as a robust result.

The pilot's primary activation is the 192-dimensional GELU output immediately
before the scalar value projection. The final graph token and mean final
crossing token are exported as controls. The kernel is a symmetric self-tuning
Gaussian using the fifth-neighbor bandwidth, followed by row normalization.

## Knot and shadow embeddings

The immediate object is a strategy on a PD shadow. A knot-type embedding claim
requires multiple diagrams per knot and stability under the allowed diagram and
label symmetries.

Candidate fixed-length summaries include diffusion spectra, diffusion-time heat
signatures, depth-stratified operator features, or occupancy-weighted barycenters
in a shared diffusion/PHATE space. Preserve the full operators as primary data;
do not reduce to a two-dimensional visualization for quantitative comparisons.

## Required controls

- untrained network with the same architecture;
- topology-only PD features;
- solved fixed-shadow MLP;
- raw policy versus MCTS policy;
- randomized policy or reward control;
- at least three training seeds;
- canonical relabeling and crossing-list permutation;
- multiple diagrams representing the same knot type;
- held-out shadow- and knot-type splits.

Coloring a PHATE plot by an invariant is exploratory evidence. Quantitative
claims require operator stability, neighbor enrichment, held-out invariant
prediction, or another preregistered statistic. Treat relationships with the
Jones-based terminal reward as potentially circular and prioritize invariants
not supplied by the reward.

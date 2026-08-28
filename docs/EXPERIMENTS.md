# Experiment Ledger

Record completed, scientifically meaningful runs here. Scheduler logs and model
artifacts remain outside Git.

## Seven-crossing fixed-shadow baseline

### Exact game

- Shadow: the single seven-crossing PD code in `config.py`
- Partial states: 2,187
- Nonterminal states: 2,059
- Terminal resolutions: 128
- Jones-polynomial-one terminals: 70
- Other terminals: 58
- Knotter moves first as player `+1`
- Forced winner: player `-1` (unknotter)
- All 14 root actions preserve the forced result

### AlphaZero run 23623670

- Date: 2026-08-25
- Commit: `bd88c8e`
- Cluster: Yale Bouchet, 2 RTX 5000 Ada GPUs
- Configuration: 20 iterations, 64 self-play episodes per iteration, 50 MCTS
  simulations, 4 training epochs
- Elapsed time: 38 minutes 54 seconds
- Optimal top-policy action across nonterminal states: 71.73%
- Mean probability mass on optimal actions: 71.52%
- Exact value-sign accuracy: 65.27%
- Root policy optimal: yes (all root actions are exact-optimal)
- Root value sign correct: no
- Certified solved verdict: **NO**

This run validates the end-to-end pipeline. It is not evidence of perfect play
and does not pass the release gate for scaling to larger shadows.

### Exact-table capacity run 23748144

- Date: 2026-08-26
- Commit: `1a035b5`
- Cluster: Yale Bouchet, 1 RTX 5000 Ada GPU
- Configuration: all 2,059 nonterminal states, 400 epochs, batch size 128,
  learning rate 0.003, seed 0
- Elapsed time: 3 minutes 49 seconds
- Optimal top-policy action: 71.49%
- Mean probability mass on optimal actions: 71.44%
- Exact value-sign accuracy: 65.42%
- Certified capacity-solved verdict: **NO**

This supervised exact-table run showed that the original architecture could not
even memorize the fixed game's complete solution, separating its capacity or
optimization failure from AlphaZero self-play and search.

### Ordered-PD capacity run 23760015

- Date: 2026-08-26
- Commit: `858684c`
- Cluster: Yale Bouchet, 1 RTX 5000 Ada GPU
- Change: concatenate the four crossing-slot embeddings in PD order before
  projection instead of summing them
- Configuration: all 2,059 nonterminal states, 400 epochs, batch size 128,
  learning rate 0.003, seed 0
- Elapsed time: 3 minutes 45 seconds
- Optimal top-policy action: 71.69%
- Mean probability mass on optimal actions: 71.44%
- Exact value-sign accuracy: 65.37%
- Certified capacity-solved verdict: **NO**

Preserving the ordered PD tuple was mathematically necessary but did not remove
the capacity ceiling. The unchanged optimal-action probability mass and nearly
flat exact metrics point to the dynamic resolution-state representation or its
use by the network as the next component to isolate.

### Direct crossing-state capacity run 23766423

- Date: 2026-08-26
- Commit: `5a785c6`
- Cluster: Yale Bouchet, 1 RTX 5000 Ada GPU
- Change: attach `unresolved`, `original`, or `switched` directly to every
  four-valent crossing node and embed that state in the network
- Configuration: all 2,059 nonterminal states, 400 epochs, batch size 128,
  learning rate 0.003, seed 0
- Elapsed time: 4 minutes 25 seconds
- Optimal top-policy action: 72.03%
- Mean probability mass on optimal actions: 71.44%
- Exact value-sign accuracy: 65.32%
- Certified capacity-solved verdict: **NO**

The explicit state was available to the network but did not remove the
collapse: optimal policy mass remained at the uniform-legal baseline and value
accuracy remained near the constant-majority baseline. A non-graph MLP over the
seven categorical crossing states is the next control needed to distinguish a
training-pipeline defect from a graph-architecture failure.

### Crossing-state MLP capacity run 23775537

- Date: 2026-08-26
- Commit: `2c40a13`
- Cluster: Yale Bouchet, 1 RTX 5000 Ada GPU
- Architecture: three-layer, width-256 MLP over the ordered seven categorical
  crossing states, with separate policy and value heads
- Configuration: all 2,059 nonterminal states, maximum 400 epochs, batch size
  128, learning rate 0.003, seed 0
- Solved at epoch: 38
- Elapsed time: 20 seconds, including independent exhaustive evaluation
- Optimal top-policy action: 100.00%
- Mean probability mass on optimal actions: 99.60%
- Exact value-sign accuracy: 100.00%
- Certified capacity-solved verdict: **YES**

This is a supervised memorization/capacity result, not self-play discovery or
generalization. It validates the exact targets, crossing-state data, batching,
losses, and optimizer. In contrast with the graph runs, it isolates the
remaining capacity failure to the graph model's representation, message
passing, or readout path.

### PD port-graph transformer capacity run 23801341

- Date: 2026-08-26
- Commit: `a616994`
- Cluster: Yale Bouchet, 1 RTX 5000 Ada GPU
- Architecture: four typed half-edge message-passing layers followed by a
  width-128 global crossing transformer; invariant to numeric arc relabeling
- Configuration: all 2,059 nonterminal states, 400 epochs, batch size 128,
  learning rate 0.003, seed 0
- Elapsed time: 2 minutes 31 seconds
- Optimal top-policy action: 72.12%
- Mean probability mass on optimal actions: 71.44%
- Exact value-sign accuracy: 65.27%
- Certified capacity-solved verdict: **NO**

The model remained at the uniform-legal policy-mass baseline and the
constant-majority value baseline for the entire run. The PD-native relations
and global attention therefore did not by themselves fix the graph-model
collapse. This result motivates an explicit state-distinguishability audit of
intermediate port and crossing embeddings before further training runs.

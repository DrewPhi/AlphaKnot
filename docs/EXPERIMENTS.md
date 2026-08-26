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

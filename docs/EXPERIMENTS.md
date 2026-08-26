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

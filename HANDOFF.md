# Handoff for next agent (written 2026-09-16)

## Branch / repo state

- Repo: `AlphaKnot/` on branch `feature/variable-crossing-prime-corpus`,
  tracking `origin/feature/variable-crossing-prime-corpus`
  (`git@github.com:DrewPhi/AlphaKnot.git`). Everything is committed and
  pushed; `git status` is clean (verify on pickup).
- `LeanKnot/` is NOT a git repo; last verified `lake build` clean on 2026-09-15.
- Paper builds: `cd paper && bibtex main && pdflatex main.tex` (x2).
  PDF is gitignored; sections + 2 figs (`figs/phate_by_{fibered,determinant}.png`) are tracked.

## What was done this session

1. Full paper draft (`paper/sections/*.tex`, ~8-9pp, 0 undefined refs):
   abstract with 3-claim split, game/methods/results, Lean formalization
   subsection (`\ref{sec:lean}`), invariant-colored PHATE pilot with 2 figures
   and operator-distance NN stats.
2. Bouchet validation job 26332105 (COMPLETED): 55/55 tests OK, 21x256
   eight-crossing terminals pass, manifest 35/35 leak-free.
3. Multi-seed shared capacity (width-192, 149,319 states): seeds 0/1/2 SOLVED
   YES at epochs 190/230/240 (jobs 24128608/26332106/26332107). G1 three-seed
   gate closed.
4. Local 9-crossing KnotInfo structural validation: 49/49 shadows, 25,088
   terminals pass (`/tmp/opencode/knotdb/validate_9x.py` + log; NOT in repo).
5. Invariant-colored PHATE (`phate_invariant_colors.py`, needs
   `--knotinfo-csv` from database_knotinfo snapshot): negative result --
   geometry tracks crossing number (30/35 NN vs 0.39), not invariants.
6. `--eval-split` held-out grading added to `variable_size_capacity_test.py`
   (+ README/methods docs); smoke job 26384443 COMPLETED in 6m22s.
7. Sage blocker documented: SnapPy 3.3.2 `jones_polynomial` raises
   `SageNotAvailable` on Bouchet, so G2-gate-2 independent comparison is
   blocked (see EXPERIMENTS.md). `max_validated_crossings` stays at 7.

## In flight (check first on pickup)

- Generalization jobs on Bouchet (train split -> eval test split,
  width-192, 300 epochs, seeds 0/1/2): **26384918, 26384919, 26384920, ALL
  COMPLETE** (~1.7-1.9h each). Train SOLVED YES @235/245/235; held-out
  15/15 NO (~71% policy). Recorded in `docs/EXPERIMENTS.md` + paper
  `results/abstract/discussion` on 2026-09-16 (commit pending).
- Next (user-approved 2026-09-16): exact supervision through 10 crossings
  (Spherogram 9_x/10_x coverage verified OK on Bouchet), self-play past the
  exact ceiling, arena + raw-vs-MCTS bar, then big PHATE + invariant coloring.
- Next proposal (user-aligned "ideal paper" arc): extend corpus to 9
  crossings via Spherogram hands (probe pending -- Spherogram name coverage
  for 9_x NOT yet verified), exact tables (3^9 states/shadow), retune,
  then full-panel PHATE + invariant enrichment.

## Cluster access notes

- Bouchet login: `dss227@bouchet.ycrc.yale.edu`, key `~/.ssh/yale`, via
  multiplex socket `~/Projects/alphaknots/.codex-bouchet-24h.sock`
  (ControlPersist may have expired -- re-establish with Duo if dead).
- Remote code: `~/AlphaKnot` (plain copy, rsynced; remote git is stale at
  72cd58d -- use rsync, not git pull, to sync; exclude .git/checkpoints/*.out).
- Raw KnotInfo 1.7M-PD file lives ONLY in `/tmp/opencode/knotdb/` (ephemeral).
- At end of last session the ssh tool channel was flaky (empty output on
  remote commands though socket master reported alive). If remote commands
  fail, re-check the socket with `ssh -S <sock> -O check`.

## Sources of truth (per AGENTS.md)

`exact_solver.py` > `evaluate_exact.py` > arena > random-play. Never claim
solved/generalized from arena alone. Capacity uses exact labels -- say so.

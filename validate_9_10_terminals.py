#!/usr/bin/env python3
"""Validate the Jones terminal classifier on nine- and ten-crossing diagrams.

For each 9- or 10-crossing catalog record: enumerate all 2^n resolved states,
check structural PD validity (labels 1..2n twice each, cyclic under-strand
neighbors), check Jones determinism and canonical-serialization invariance,
and report the Jones=1 count.

This does NOT raise config.max_validated_crossings by itself. A production
bound change additionally requires an independent SnapPy/KnotInfo table
comparison (see docs/ROADMAP.md G2 gate 2); this script is the structural +
determinism precondition for that comparison.
"""

import argparse
import json
import multiprocessing
from collections import Counter

from corpus_manifest import build_manifest
from exact_solver import ExactSolver
from knot_invariants import jones_is_one, normalized_jones_in_a
from pd_code_utils import canonicalize_pd_code, crossing_sign


def validate_shadow(job):
    name, pd_code = job
    n = len(pd_code)
    solver = ExactSolver(pd_code)
    terminals = list(solver.all_states(terminal=True))
    assert len(terminals) == 2 ** n
    unknots = 0
    for state in terminals:
        resolved = solver.resolved_pd_code(state)
        labels = [int(x) for crossing in resolved for x in crossing]
        occurrences = Counter(labels)
        assert set(occurrences) == set(range(1, 2 * len(resolved) + 1))
        assert all(count == 2 for count in occurrences.values())
        n_edges = 2 * len(resolved)
        for crossing in resolved:
            a, _, c, _ = map(int, crossing)
            assert (a % n_edges) + 1 == c or (c % n_edges) + 1 == a
            crossing_sign(crossing, n_edges)
        first = jones_is_one(resolved)
        assert first == jones_is_one([list(c) for c in resolved])
        # Canonical relabelings must not change the invariant.
        canonical = [list(c) for c in canonicalize_pd_code(resolved).pd_code]
        assert jones_is_one(canonical) == first
        unknots += first
    # Determinism spot-check on the source diagram itself.
    assert normalized_jones_in_a(solver.pd_code) == normalized_jones_in_a(
        [list(c) for c in solver.pd_code]
    )
    return {"knot": name, "terminals": len(terminals), "jones_one": unknots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossings", type=int, choices=(9, 10), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    manifest = [r for r in build_manifest(3, 10) if r["crossings"] == args.crossings]
    assert len(manifest) == {9: 49, 10: 165}[args.crossings]
    jobs = sorted(
        (r["knot"], r["canonical_pd"]) for r in manifest
    )
    with multiprocessing.Pool(args.workers) as pool:
        results = pool.map(validate_shadow, jobs)
    results.sort(key=lambda row: row["knot"])
    total = sum(r["terminals"] for r in results)
    print(f"Validated {len(results)} {args.crossings}-crossing shadows, "
          f"{total} terminals")
    for row in results:
        print(f"{row['knot']:>7} terminals={row['terminals']} "
              f"jones_one={row['jones_one']}")
    print("NOTE: independent SnapPy/KnotInfo comparison still required before "
          "raising max_validated_crossings.")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

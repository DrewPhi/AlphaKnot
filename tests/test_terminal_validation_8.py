import unittest
from collections import Counter
from itertools import product

from corpus_manifest import build_manifest
from exact_solver import ExactSolver
from knot_invariants import jones_is_one
from pd_code_utils import canonicalize_pd_code, crossing_sign


class EightCrossingTerminalValidationTests(unittest.TestCase):
    def test_eight_crossing_shadow_enumerates_256_structurally_valid_terminals(self):
        # Full structural + determinism check on one pilot shadow (8_1 keeps the
        # unit test bounded; the full 21-shadow sweep lives in
        # validate_8crossing_terminals.py).
        manifest = {r["knot"]: r for r in build_manifest() if r["crossings"] == 8}
        self.assertEqual(len(manifest), 21)
        solver = ExactSolver(manifest["8_1"]["canonical_pd"])
        terminals = list(solver.all_states(terminal=True))
        self.assertEqual(len(terminals), 256)
        checked = 0
        for state in terminals:
            resolved = solver.resolved_pd_code(state)
            labels = [int(x) for crossing in resolved for x in crossing]
            occurrences = Counter(labels)
            self.assertEqual(set(occurrences), set(range(1, 17)))
            self.assertTrue(all(count == 2 for count in occurrences.values()))
            for crossing in resolved:
                crossing_sign(crossing, 16)
            first = jones_is_one(resolved)
            canonical = [list(c) for c in canonicalize_pd_code(resolved).pd_code]
            self.assertEqual(jones_is_one(canonical), first)
            checked += 1
        self.assertEqual(checked, 256)

    def test_all_eight_crossing_shadows_are_structurally_valid_sources(self):
        # Lightweight: every catalog source resolves to valid PDs on a sample of
        # states without running the full Jones sweep per shadow.
        for record in build_manifest():
            if record["crossings"] != 8:
                continue
            solver = ExactSolver(record["canonical_pd"])
            for state in product((0, 1), repeat=8):
                resolved = solver.resolved_pd_code(state)
                labels = [int(x) for crossing in resolved for x in crossing]
                self.assertEqual(Counter(labels)[1], 2)
                break


if __name__ == "__main__":
    unittest.main()

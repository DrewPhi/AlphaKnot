import unittest

from corpus_manifest import (
    MANIFEST_VERSION,
    SPLIT_BY_KNOT,
    assert_no_leakage,
    build_manifest,
    exact_table_checksum,
    manifest_checksum,
    records_for_split,
)
from prime_knot_corpus import PRIME_KNOT_COUNTS, PRIME_KNOT_PD_CODES


class CorpusManifestTests(unittest.TestCase):
    def test_manifest_covers_full_catalog(self):
        manifest = build_manifest()
        self.assertEqual(len(manifest), 35)
        self.assertEqual({r["manifest_version"] for r in manifest}, {MANIFEST_VERSION})
        self.assertEqual(
            {r["knot"] for r in manifest},
            {f"{c}_{i}" for c, n in PRIME_KNOT_COUNTS.items() for i in range(1, n + 1)},
        )

    def test_splits_are_frozen_disjoint_and_cover_catalog(self):
        manifest = build_manifest()
        by_knot = {r["knot"]: r["split"] for r in manifest}
        self.assertEqual(by_knot, SPLIT_BY_KNOT)
        self.assertTrue(all(s in ("train", "validation", "test") for s in by_knot.values()))
        for split in ("train", "validation", "test"):
            self.assertTrue(records_for_split(split))
        total = sum(len(records_for_split(s)) for s in ("train", "validation", "test"))
        self.assertEqual(total, 35)

    def test_no_shadow_leakage_and_stable_checksum(self):
        manifest = build_manifest()
        assert_no_leakage(manifest)
        # Shadow-level dedup: all 35 table diagrams are distinct shadows here.
        self.assertEqual(len({r["equivalence_group"] for r in manifest}), 35)
        self.assertEqual(manifest_checksum(manifest), manifest_checksum(build_manifest()))
        # G1.1 gate: every record advertises exact-table availability.
        self.assertTrue(all(r["exact_table_available"] for r in manifest))

    def test_exact_table_checksum_stable_on_small_shadow(self):
        # 3_1 has only 8 terminals; full 8-crossing checksums live in
        # validate_8crossing_terminals.py to keep unit tests fast.
        first = exact_table_checksum(PRIME_KNOT_PD_CODES["3_1"])
        second = exact_table_checksum(PRIME_KNOT_PD_CODES["3_1"])
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

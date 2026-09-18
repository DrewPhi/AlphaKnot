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
from prime_knot_corpus import PRIME_KNOT_COUNTS, PRIME_KNOT_PD_CODES, corpus_records


class CorpusManifestTests(unittest.TestCase):
    def test_manifest_covers_full_catalog(self):
        manifest = build_manifest()
        self.assertEqual(len(manifest), 35)
        self.assertEqual({r["manifest_version"] for r in manifest}, {MANIFEST_VERSION})
        self.assertEqual(
            {r["knot"] for r in manifest},
            {name for name, _ in corpus_records()},
        )

    def test_splits_are_frozen_disjoint_and_cover_catalog(self):
        manifest = build_manifest()
        by_knot = {r["knot"]: r["split"] for r in manifest}
        self.assertEqual(
            by_knot, {k: v for k, v in SPLIT_BY_KNOT.items() if k in by_knot}
        )
        self.assertTrue(all(s in ("train", "validation", "test") for s in by_knot.values()))
        for split in ("train", "validation", "test"):
            self.assertTrue(records_for_split(split))
        total = sum(len(records_for_split(s)) for s in ("train", "validation", "test"))
        self.assertEqual(total, 35)

    def test_extended_catalog_through_ten_crossings(self):
        manifest = build_manifest(3, 10)
        self.assertEqual(len(manifest), 249)
        by_knot = {r["knot"]: r["split"] for r in manifest}
        self.assertEqual(by_knot, SPLIT_BY_KNOT)
        assert_no_leakage(manifest)
        self.assertEqual(len({r["equivalence_group"] for r in manifest}), 249)
        self.assertEqual(
            {r["manifest_version"] for r in manifest}, {MANIFEST_VERSION}
        )
        counts = {}
        for r in manifest:
            counts[r["crossings"]] = counts.get(r["crossings"], 0) + 1
        self.assertEqual(
            counts, {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 165}
        )
        # Every crossing count stays represented in train.
        train_crossings = {
            r["crossings"] for r in manifest if r["split"] == "train"
        }
        self.assertEqual(train_crossings, {3, 4, 5, 6, 7, 8, 9, 10})
        total = sum(
            len(records_for_split(s, 3, 10))
            for s in ("train", "validation", "test")
        )
        self.assertEqual(total, 249)

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

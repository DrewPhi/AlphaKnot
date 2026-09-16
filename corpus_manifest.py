"""Versioned shadow manifest and frozen splits for the prime-knot catalog.

This implements ROADMAP G1.1/G2 gate 3: one record per shadow with a stable
``shadow_id``, source diagram, canonical PD, canonicalization version,
equivalence group, frozen split, and exact-table availability. The per-shadow
exact-table checksum is provided on demand by ``exact_table_checksum`` rather
than stored in every record, keeping ``build_manifest`` fast (a full checksum
sweep costs seconds per 8-crossing shadow).

The catalog holds one Spherogram table diagram per prime knot type (35 records,
3-8 crossings). It is not an enumeration of all shadows. Splits are
knot-type-disjoint: canonical relabelings / crossing-list permutations of one
object must never leak across a split.
"""

import hashlib
import json

from pd_code_utils import PD_CANONICALIZATION_VERSION, canonicalize_pd_code
from prime_knot_corpus import corpus_records

MANIFEST_VERSION = "prime-catalog-v1"

# Frozen knot-type-disjoint splits. Singletons (3_1, 4_1) stay in train so every
# crossing count remains represented during training.
SPLIT_BY_KNOT = {
    "3_1": "train",
    "4_1": "train",
    "5_1": "train",
    "5_2": "test",
    "6_1": "train",
    "6_2": "train",
    "6_3": "validation",
    "7_1": "train",
    "7_2": "train",
    "7_3": "train",
    "7_4": "train",
    "7_5": "train",
    "7_6": "validation",
    "7_7": "test",
    "8_1": "train",
    "8_2": "train",
    "8_3": "train",
    "8_4": "train",
    "8_5": "train",
    "8_6": "train",
    "8_7": "train",
    "8_8": "train",
    "8_9": "train",
    "8_10": "train",
    "8_11": "train",
    "8_12": "train",
    "8_13": "train",
    "8_14": "train",
    "8_15": "train",
    "8_16": "validation",
    "8_17": "validation",
    "8_18": "validation",
    "8_19": "test",
    "8_20": "test",
    "8_21": "test",
}


def _shadow_id(canonical_pd):
    payload = json.dumps(canonical_pd, sort_keys=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_manifest(min_crossings=3, max_crossings=8):
    """Return a list of manifest records in stable table order."""
    records = corpus_records(min_crossings, max_crossings)
    manifest = []
    seen_canonical = {}
    for name, pd_code in records:
        canonical = canonicalize_pd_code(pd_code)
        canonical_lists = [list(crossing) for crossing in canonical.pd_code]
        shadow_id = _shadow_id(canonical_lists)
        # Shadow-level dedup: identical canonical PDs share an equivalence group.
        group = seen_canonical.setdefault(tuple(canonical.pd_code), shadow_id)
        if name not in SPLIT_BY_KNOT:
            raise ValueError(f"No frozen split assigned for {name}")
        manifest.append(
            {
                "shadow_id": shadow_id,
                "knot": name,
                "source": f"spherogram-table-{name}",
                "crossings": len(pd_code),
                "canonical_pd": canonical_lists,
                "canonicalization_version": PD_CANONICALIZATION_VERSION,
                "equivalence_group": group,
                "split": SPLIT_BY_KNOT[name],
                "manifest_version": MANIFEST_VERSION,
                "exact_table_available": True,
            }
        )
    return manifest


def manifest_checksum(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_no_leakage(manifest):
    """Equivalent canonical objects must share one split."""
    group_splits = {}
    for record in manifest:
        group = record["equivalence_group"]
        previous = group_splits.setdefault(group, record["split"])
        if previous != record["split"]:
            raise ValueError(
                f"Equivalence group {group} leaks across splits: "
                f"{previous} vs {record['split']}"
            )


def records_for_split(split, min_crossings=3, max_crossings=8):
    """Return (name, pd_code) records for one frozen split."""
    wanted = {
        record["knot"]
        for record in build_manifest(min_crossings, max_crossings)
        if record["split"] == split
    }
    return [
        (name, pd_code)
        for name, pd_code in corpus_records(min_crossings, max_crossings)
        if name in wanted
    ]


def exact_table_checksum(pd_code):
    """Checksum over exhaustive terminal Jones outcomes for one shadow.

    Enumerates all 2^n resolved states via ExactSolver.is_unknot. This is
    trivial at n<=8 (256 terminals) and intentionally expensive beyond that,
    so callers should only use it for the validated regime.
    """
    from exact_solver import ExactSolver  # deferred: heavy import, avoids cycles

    solver = ExactSolver(pd_code)
    outcomes = [1 if solver.is_unknot(state) else 0 for state in solver.all_states(terminal=True)]
    payload = json.dumps(
        {"canonical": [list(c) for c in solver.pd_code], "terminals": outcomes},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

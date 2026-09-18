"""Versioned shadow manifest and frozen splits for the prime-knot catalog.

This implements ROADMAP G1.1/G2 gate 3: one record per shadow with a stable
``shadow_id``, source diagram, canonical PD, canonicalization version,
equivalence group, frozen split, and exact-table availability. The per-shadow
exact-table checksum is provided on demand by ``exact_table_checksum`` rather
than stored in every record, keeping ``build_manifest`` fast (a full checksum
sweep costs seconds per 8-crossing shadow).

The catalog holds one Spherogram table diagram per prime knot type (249 records,
3-10 crossings; the 3-8 legacy range holds 35). It is not an enumeration of
all shadows. Splits are
knot-type-disjoint: canonical relabelings / crossing-list permutations of one
object must never leak across a split.
"""

import hashlib
import json

from pd_code_utils import PD_CANONICALIZATION_VERSION, canonicalize_pd_code
from prime_knot_corpus import corpus_records

MANIFEST_VERSION = "prime-catalog-v2"

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
    "9_1": "train",
    "9_2": "train",
    "9_3": "train",
    "9_4": "train",
    "9_5": "train",
    "9_6": "train",
    "9_7": "train",
    "9_8": "train",
    "9_9": "train",
    "9_10": "train",
    "9_11": "train",
    "9_12": "train",
    "9_13": "train",
    "9_14": "train",
    "9_15": "train",
    "9_16": "train",
    "9_17": "train",
    "9_18": "train",
    "9_19": "train",
    "9_20": "train",
    "9_21": "train",
    "9_22": "train",
    "9_23": "train",
    "9_24": "train",
    "9_25": "train",
    "9_26": "train",
    "9_27": "train",
    "9_28": "train",
    "9_29": "train",
    "9_30": "train",
    "9_31": "train",
    "9_32": "train",
    "9_33": "train",
    "9_34": "train",
    "9_35": "train",
    "9_36": "train",
    "9_37": "train",
    "9_38": "train",
    "9_39": "train",
    "9_40": "train",
    "9_41": "train",
    "9_42": "train",
    "9_43": "train",
    "9_44": "validation",
    "9_45": "validation",
    "9_46": "validation",
    "9_47": "test",
    "9_48": "test",
    "9_49": "test",
    "10_1": "train",
    "10_2": "train",
    "10_3": "train",
    "10_4": "train",
    "10_5": "train",
    "10_6": "train",
    "10_7": "train",
    "10_8": "train",
    "10_9": "train",
    "10_10": "train",
    "10_11": "train",
    "10_12": "train",
    "10_13": "train",
    "10_14": "train",
    "10_15": "train",
    "10_16": "train",
    "10_17": "train",
    "10_18": "train",
    "10_19": "train",
    "10_20": "train",
    "10_21": "train",
    "10_22": "train",
    "10_23": "train",
    "10_24": "train",
    "10_25": "train",
    "10_26": "train",
    "10_27": "train",
    "10_28": "train",
    "10_29": "train",
    "10_30": "train",
    "10_31": "train",
    "10_32": "train",
    "10_33": "train",
    "10_34": "train",
    "10_35": "train",
    "10_36": "train",
    "10_37": "train",
    "10_38": "train",
    "10_39": "train",
    "10_40": "train",
    "10_41": "train",
    "10_42": "train",
    "10_43": "train",
    "10_44": "train",
    "10_45": "train",
    "10_46": "train",
    "10_47": "train",
    "10_48": "train",
    "10_49": "train",
    "10_50": "train",
    "10_51": "train",
    "10_52": "train",
    "10_53": "train",
    "10_54": "train",
    "10_55": "train",
    "10_56": "train",
    "10_57": "train",
    "10_58": "train",
    "10_59": "train",
    "10_60": "train",
    "10_61": "train",
    "10_62": "train",
    "10_63": "train",
    "10_64": "train",
    "10_65": "train",
    "10_66": "train",
    "10_67": "train",
    "10_68": "train",
    "10_69": "train",
    "10_70": "train",
    "10_71": "train",
    "10_72": "train",
    "10_73": "train",
    "10_74": "train",
    "10_75": "train",
    "10_76": "train",
    "10_77": "train",
    "10_78": "train",
    "10_79": "train",
    "10_80": "train",
    "10_81": "train",
    "10_82": "train",
    "10_83": "train",
    "10_84": "train",
    "10_85": "train",
    "10_86": "train",
    "10_87": "train",
    "10_88": "train",
    "10_89": "train",
    "10_90": "train",
    "10_91": "train",
    "10_92": "train",
    "10_93": "train",
    "10_94": "train",
    "10_95": "train",
    "10_96": "train",
    "10_97": "train",
    "10_98": "train",
    "10_99": "train",
    "10_100": "train",
    "10_101": "train",
    "10_102": "train",
    "10_103": "train",
    "10_104": "train",
    "10_105": "train",
    "10_106": "train",
    "10_107": "train",
    "10_108": "train",
    "10_109": "train",
    "10_110": "train",
    "10_111": "train",
    "10_112": "train",
    "10_113": "train",
    "10_114": "train",
    "10_115": "train",
    "10_116": "train",
    "10_117": "train",
    "10_118": "train",
    "10_119": "train",
    "10_120": "train",
    "10_121": "train",
    "10_122": "train",
    "10_123": "train",
    "10_124": "train",
    "10_125": "train",
    "10_126": "train",
    "10_127": "train",
    "10_128": "train",
    "10_129": "train",
    "10_130": "train",
    "10_131": "train",
    "10_132": "train",
    "10_133": "train",
    "10_134": "train",
    "10_135": "train",
    "10_136": "train",
    "10_137": "train",
    "10_138": "train",
    "10_139": "train",
    "10_140": "train",
    "10_141": "train",
    "10_142": "train",
    "10_143": "train",
    "10_144": "train",
    "10_145": "train",
    "10_146": "train",
    "10_147": "train",
    "10_148": "train",
    "10_149": "train",
    "10_150": "train",
    "10_151": "train",
    "10_152": "train",
    "10_153": "train",
    "10_154": "train",
    "10_155": "train",
    "10_156": "validation",
    "10_157": "validation",
    "10_158": "validation",
    "10_159": "validation",
    "10_160": "validation",
    "10_161": "test",
    "10_162": "test",
    "10_163": "test",
    "10_164": "test",
    "10_165": "test",
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

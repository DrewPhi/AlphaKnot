"""Versioned standard PD diagrams for prime knots through eight crossings.

The records were exported from Spherogram 2.4.1 with
``Link(name).PD_code(min_strand_index=1)``.  Keeping the table in Git makes
training reproducible and avoids a runtime dependency on SnapPy/Spherogram.

These are standard table *diagrams*: one PD code per prime knot type.  They are
not an enumeration of all shadows or all diagrams representing each knot.
"""

from collections import Counter

from pd_code_utils import crossing_sign


PRIME_KNOT_PD_CODES = {
    "3_1": [[6, 3, 1, 4], [4, 1, 5, 2], [2, 5, 3, 6]],
    "4_1": [[8, 5, 1, 6], [4, 1, 5, 2], [2, 8, 3, 7], [6, 4, 7, 3]],
    "5_1": [[10, 5, 1, 6], [6, 1, 7, 2], [2, 7, 3, 8], [8, 3, 9, 4], [4, 9, 5, 10]],
    "5_2": [[5, 1, 6, 10], [1, 7, 2, 6], [9, 3, 10, 2], [3, 9, 4, 8], [7, 5, 8, 4]],
    "6_1": [[7, 12, 8, 1], [1, 6, 2, 7], [11, 3, 12, 2], [3, 11, 4, 10], [9, 5, 10, 4], [5, 9, 6, 8]],
    "6_2": [[12, 8, 1, 7], [8, 2, 9, 1], [2, 10, 3, 9], [6, 4, 7, 3], [4, 11, 5, 12], [10, 5, 11, 6]],
    "6_3": [[9, 12, 10, 1], [1, 5, 2, 4], [7, 3, 8, 2], [3, 9, 4, 8], [5, 10, 6, 11], [11, 6, 12, 7]],
    "7_1": [[14, 7, 1, 8], [8, 1, 9, 2], [2, 9, 3, 10], [10, 3, 11, 4], [4, 11, 5, 12], [12, 5, 13, 6], [6, 13, 7, 14]],
    "7_2": [[14, 11, 1, 12], [10, 1, 11, 2], [2, 9, 3, 10], [8, 3, 9, 4], [4, 7, 5, 8], [12, 5, 13, 6], [6, 13, 7, 14]],
    "7_3": [[14, 9, 1, 10], [8, 1, 9, 2], [2, 7, 3, 8], [10, 3, 11, 4], [4, 11, 5, 12], [12, 5, 13, 6], [6, 13, 7, 14]],
    "7_4": [[14, 8, 1, 7], [6, 2, 7, 1], [2, 12, 3, 11], [10, 4, 11, 3], [4, 10, 5, 9], [12, 6, 13, 5], [8, 14, 9, 13]],
    "7_5": [[14, 5, 1, 6], [4, 1, 5, 2], [2, 9, 3, 10], [10, 3, 11, 4], [6, 11, 7, 12], [12, 7, 13, 8], [8, 13, 9, 14]],
    "7_6": [[9, 14, 10, 1], [1, 8, 2, 9], [13, 3, 14, 2], [3, 13, 4, 12], [7, 4, 8, 5], [5, 10, 6, 11], [11, 6, 12, 7]],
    "7_7": [[7, 14, 8, 1], [1, 6, 2, 7], [11, 2, 12, 3], [3, 10, 4, 11], [13, 5, 14, 4], [5, 9, 6, 8], [9, 13, 10, 12]],
    "8_1": [[16, 14, 1, 13], [12, 2, 13, 1], [2, 12, 3, 11], [10, 4, 11, 3], [4, 10, 5, 9], [8, 6, 9, 5], [6, 15, 7, 16], [14, 7, 15, 8]],
    "8_2": [[9, 16, 10, 1], [1, 10, 2, 11], [11, 2, 12, 3], [3, 12, 4, 13], [13, 4, 14, 5], [5, 8, 6, 9], [15, 7, 16, 6], [7, 15, 8, 14]],
    "8_3": [[16, 12, 1, 11], [10, 2, 11, 1], [2, 10, 3, 9], [8, 4, 9, 3], [4, 15, 5, 16], [14, 5, 15, 6], [6, 13, 7, 14], [12, 7, 13, 8]],
    "8_4": [[16, 8, 1, 7], [8, 2, 9, 1], [2, 10, 3, 9], [14, 3, 15, 4], [4, 13, 5, 14], [12, 5, 13, 6], [6, 11, 7, 12], [10, 16, 11, 15]],
    "8_5": [[11, 1, 12, 16], [1, 13, 2, 12], [13, 3, 14, 2], [3, 9, 4, 8], [9, 5, 10, 4], [5, 11, 6, 10], [15, 6, 16, 7], [7, 14, 8, 15]],
    "8_6": [[11, 16, 12, 1], [1, 12, 2, 13], [13, 2, 14, 3], [3, 10, 4, 11], [9, 4, 10, 5], [5, 8, 6, 9], [15, 7, 16, 6], [7, 15, 8, 14]],
    "8_7": [[16, 7, 1, 8], [8, 1, 9, 2], [2, 9, 3, 10], [10, 3, 11, 4], [4, 14, 5, 13], [14, 6, 15, 5], [6, 11, 7, 12], [12, 16, 13, 15]],
    "8_8": [[5, 16, 6, 1], [1, 6, 2, 7], [11, 3, 12, 2], [3, 13, 4, 12], [7, 4, 8, 5], [15, 9, 16, 8], [9, 15, 10, 14], [13, 11, 14, 10]],
    "8_9": [[16, 7, 1, 8], [8, 1, 9, 2], [2, 9, 3, 10], [14, 4, 15, 3], [4, 12, 5, 11], [12, 6, 13, 5], [6, 14, 7, 13], [10, 15, 11, 16]],
    "8_10": [[16, 10, 1, 9], [4, 1, 5, 2], [2, 13, 3, 14], [14, 3, 15, 4], [10, 6, 11, 5], [6, 12, 7, 11], [12, 8, 13, 7], [8, 16, 9, 15]],
    "8_11": [[11, 16, 12, 1], [1, 10, 2, 11], [9, 2, 10, 3], [3, 12, 4, 13], [13, 4, 14, 5], [5, 8, 6, 9], [15, 7, 16, 6], [7, 15, 8, 14]],
    "8_12": [[16, 12, 1, 11], [10, 2, 11, 1], [2, 8, 3, 7], [6, 4, 7, 3], [4, 15, 5, 16], [14, 5, 15, 6], [8, 13, 9, 14], [12, 9, 13, 10]],
    "8_13": [[16, 7, 1, 8], [8, 1, 9, 2], [2, 14, 3, 13], [12, 4, 13, 3], [4, 12, 5, 11], [14, 6, 15, 5], [6, 9, 7, 10], [10, 16, 11, 15]],
    "8_14": [[9, 1, 10, 16], [1, 11, 2, 10], [13, 3, 14, 2], [3, 7, 4, 6], [15, 4, 16, 5], [5, 14, 6, 15], [7, 13, 8, 12], [11, 9, 12, 8]],
    "8_15": [[11, 16, 12, 1], [1, 10, 2, 11], [5, 2, 6, 3], [3, 14, 4, 15], [15, 4, 16, 5], [9, 6, 10, 7], [7, 12, 8, 13], [13, 8, 14, 9]],
    "8_16": [[16, 12, 1, 11], [6, 1, 7, 2], [2, 9, 3, 10], [14, 4, 15, 3], [4, 16, 5, 15], [10, 5, 11, 6], [12, 8, 13, 7], [8, 14, 9, 13]],
    "8_17": [[16, 7, 1, 8], [12, 2, 13, 1], [2, 10, 3, 9], [14, 3, 15, 4], [4, 15, 5, 16], [10, 6, 11, 5], [6, 12, 7, 11], [8, 13, 9, 14]],
    "8_18": [[11, 16, 12, 1], [1, 7, 2, 6], [13, 3, 14, 2], [3, 8, 4, 9], [15, 4, 16, 5], [5, 11, 6, 10], [7, 12, 8, 13], [9, 15, 10, 14]],
    "8_19": [[16, 6, 1, 5], [6, 2, 7, 1], [11, 3, 12, 2], [3, 15, 4, 14], [4, 10, 5, 9], [12, 8, 13, 7], [8, 14, 9, 13], [15, 11, 16, 10]],
    "8_20": [[16, 7, 1, 8], [1, 12, 2, 13], [9, 2, 10, 3], [14, 3, 15, 4], [4, 15, 5, 16], [10, 6, 11, 5], [6, 12, 7, 11], [13, 9, 14, 8]],
    "8_21": [[7, 1, 8, 16], [1, 12, 2, 13], [2, 10, 3, 9], [14, 3, 15, 4], [4, 15, 5, 16], [5, 10, 6, 11], [11, 6, 12, 7], [8, 13, 9, 14]],
}


PRIME_KNOT_COUNTS = {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21}


def _name_key(name):
    crossings, index = name.split("_")
    return int(crossings), int(index)


def validate_corpus(corpus=PRIME_KNOT_PD_CODES, min_crossings=3, max_crossings=8):
    """Validate names and oriented one-component PD conventions."""
    expected = {
        f"{crossings}_{index}"
        for crossings, count in PRIME_KNOT_COUNTS.items()
        if min_crossings <= crossings <= max_crossings
        for index in range(1, count + 1)
    }
    selected = {
        name: pd_code
        for name, pd_code in corpus.items()
        if min_crossings <= len(pd_code) <= max_crossings
    }
    if set(selected) != expected:
        raise ValueError(f"Corpus names do not match the prime-knot table: {sorted(selected)}")

    for name, pd_code in selected.items():
        crossings = int(name.split("_")[0])
        if len(pd_code) != crossings:
            raise ValueError(f"{name} has {len(pd_code)} crossings")
        labels = [int(label) for crossing in pd_code for label in crossing]
        occurrences = Counter(labels)
        if set(occurrences) != set(range(1, 2 * crossings + 1)):
            raise ValueError(f"{name} does not use labels 1..{2 * crossings}")
        if any(count != 2 for count in occurrences.values()):
            raise ValueError(f"{name} does not use every arc label exactly twice")
        for crossing in pd_code:
            crossing_sign(crossing, 2 * crossings)


def corpus_records(min_crossings=3, max_crossings=8):
    """Return defensive copies in stable crossing-number/table order."""
    validate_corpus(
        min_crossings=min_crossings,
        max_crossings=max_crossings,
    )
    return [
        (name, [list(crossing) for crossing in PRIME_KNOT_PD_CODES[name]])
        for name in sorted(PRIME_KNOT_PD_CODES, key=_name_key)
        if min_crossings <= len(PRIME_KNOT_PD_CODES[name]) <= max_crossings
    ]

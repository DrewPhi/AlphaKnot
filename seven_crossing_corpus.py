"""Reproducible PD-code corpus for the seven prime seven-crossing knots.

The records were exported from SnapPy/Spherogram 3.1.1 using
``Link(name).PD_code(min_strand_index=1)``.  They are stored locally so an
experiment does not depend on SnapPy's table access at training time.
"""

from collections import Counter

from pd_code_utils import crossing_sign


SEVEN_CROSSING_PD_CODES = {
    "7_1": [
        [14, 7, 1, 8], [8, 1, 9, 2], [2, 9, 3, 10],
        [10, 3, 11, 4], [4, 11, 5, 12], [12, 5, 13, 6],
        [6, 13, 7, 14],
    ],
    "7_2": [
        [14, 11, 1, 12], [10, 1, 11, 2], [2, 9, 3, 10],
        [8, 3, 9, 4], [4, 7, 5, 8], [12, 5, 13, 6],
        [6, 13, 7, 14],
    ],
    "7_3": [
        [14, 9, 1, 10], [8, 1, 9, 2], [2, 7, 3, 8],
        [10, 3, 11, 4], [4, 11, 5, 12], [12, 5, 13, 6],
        [6, 13, 7, 14],
    ],
    "7_4": [
        [14, 8, 1, 7], [6, 2, 7, 1], [2, 12, 3, 11],
        [10, 4, 11, 3], [4, 10, 5, 9], [12, 6, 13, 5],
        [8, 14, 9, 13],
    ],
    "7_5": [
        [14, 5, 1, 6], [4, 1, 5, 2], [2, 9, 3, 10],
        [10, 3, 11, 4], [6, 11, 7, 12], [12, 7, 13, 8],
        [8, 13, 9, 14],
    ],
    "7_6": [
        [9, 14, 10, 1], [1, 8, 2, 9], [13, 3, 14, 2],
        [3, 13, 4, 12], [7, 4, 8, 5], [5, 10, 6, 11],
        [11, 6, 12, 7],
    ],
    "7_7": [
        [7, 14, 8, 1], [1, 6, 2, 7], [11, 2, 12, 3],
        [3, 10, 4, 11], [13, 5, 14, 4], [5, 9, 6, 8],
        [9, 13, 10, 12],
    ],
}


def validate_corpus(corpus=SEVEN_CROSSING_PD_CODES):
    """Validate the assumptions required by AlphaKnot's exact solver."""
    expected_names = {f"7_{index}" for index in range(1, 8)}
    if set(corpus) != expected_names:
        raise ValueError("Corpus must contain exactly 7_1 through 7_7")

    for name, pd_code in corpus.items():
        if len(pd_code) != 7:
            raise ValueError(f"{name} does not have seven crossings")
        labels = [int(label) for crossing in pd_code for label in crossing]
        occurrences = Counter(labels)
        if set(occurrences) != set(range(1, 15)):
            raise ValueError(f"{name} does not use consecutive labels 1..14")
        if any(count != 2 for count in occurrences.values()):
            raise ValueError(f"{name} does not use every arc label exactly twice")
        for crossing in pd_code:
            crossing_sign(crossing, 14)


def corpus_records():
    """Return defensive copies in stable Rolfsen-table order."""
    validate_corpus()
    return [
        (name, [list(crossing) for crossing in SEVEN_CROSSING_PD_CODES[name]])
        for name in sorted(SEVEN_CROSSING_PD_CODES, key=lambda item: int(item[2:]))
    ]

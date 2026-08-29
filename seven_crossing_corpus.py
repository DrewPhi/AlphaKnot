"""Compatibility view of the versioned prime-knot PD corpus."""

from prime_knot_corpus import PRIME_KNOT_PD_CODES, validate_corpus as validate_prime_corpus


SEVEN_CROSSING_PD_CODES = {
    name: pd_code
    for name, pd_code in PRIME_KNOT_PD_CODES.items()
    if name.startswith("7_")
}


def validate_corpus(corpus=SEVEN_CROSSING_PD_CODES):
    """Validate the assumptions required by AlphaKnot's exact solver."""
    validate_prime_corpus(corpus, min_crossings=7, max_crossings=7)


def corpus_records():
    """Return defensive copies in stable Rolfsen-table order."""
    validate_corpus()
    return [
        (name, [list(crossing) for crossing in SEVEN_CROSSING_PD_CODES[name]])
        for name in sorted(SEVEN_CROSSING_PD_CODES, key=lambda item: int(item[2:]))
    ]

"""Small-knot invariants used by the terminal game evaluator.

The implementation is intentionally exhaustive: at the configured maximum of
seven crossings, a Kauffman-bracket evaluation has only 128 smoothing states.
It avoids the optional Sage dependency used by Spherogram's polynomial methods.
"""

from functools import lru_cache
from itertools import product

from pd_code_utils import crossing_sign


def _poly_multiply(left, right):
    result = {}
    for left_power, left_coefficient in left.items():
        for right_power, right_coefficient in right.items():
            power = left_power + right_power
            result[power] = (
                result.get(power, 0)
                + left_coefficient * right_coefficient
            )
    return {power: coefficient for power, coefficient in result.items()
            if coefficient}


def _poly_power(polynomial, exponent):
    result = {0: 1}
    for _ in range(exponent):
        result = _poly_multiply(result, polynomial)
    return result


@lru_cache(maxsize=4096)
def _normalized_jones_cached(pd_code):
    crossings = tuple(tuple(map(int, crossing)) for crossing in pd_code)
    if not crossings:
        return ((0, 1),)

    n_edges = max(label for crossing in crossings for label in crossing)
    half_edges = tuple(
        (crossing_index, position)
        for crossing_index in range(len(crossings))
        for position in range(4)
    )
    total = {}

    for smoothing_state in product((0, 1), repeat=len(crossings)):
        parent = {half_edge: half_edge for half_edge in half_edges}

        def find(item):
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(left, right):
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        label_endpoint = {}
        for crossing_index, crossing in enumerate(crossings):
            for position, label in enumerate(crossing):
                endpoint = (crossing_index, position)
                if label in label_endpoint:
                    union(endpoint, label_endpoint[label])
                else:
                    label_endpoint[label] = endpoint

        if any(
            sum(
                current_label == label
                for crossing in crossings
                for current_label in crossing
            ) != 2
            for label in label_endpoint
        ):
            raise ValueError("Every PD edge label must occur exactly twice")

        for crossing_index, smoothing in enumerate(smoothing_state):
            # With tuple positions starting at the incoming under-edge, these
            # are the A and B smoothings used by the Kauffman bracket.
            pairs = (
                ((0, 1), (2, 3))
                if smoothing == 0
                else ((0, 3), (1, 2))
            )
            for left_position, right_position in pairs:
                union(
                    (crossing_index, left_position),
                    (crossing_index, right_position),
                )

        loops = len({find(half_edge) for half_edge in half_edges})
        a_power = sum(1 if smoothing == 0 else -1
                      for smoothing in smoothing_state)
        contribution = _poly_multiply(
            {a_power: 1},
            _poly_power({2: -1, -2: -1}, loops - 1),
        )
        for power, coefficient in contribution.items():
            total[power] = total.get(power, 0) + coefficient

    writhe = sum(crossing_sign(crossing, n_edges)
                  for crossing in crossings)
    normalized = {
        power - 3 * writhe: coefficient * ((-1) ** writhe)
        for power, coefficient in total.items()
        if coefficient
    }
    return tuple(sorted(normalized.items()))


def normalized_jones_in_a(pd_code):
    """Return the normalized Jones polynomial as ``{A exponent: coefficient}``."""
    key = tuple(tuple(map(int, crossing)) for crossing in pd_code)
    return dict(_normalized_jones_cached(key))


def jones_is_one(pd_code):
    """Whether the normalized Jones polynomial equals the unknot's value."""
    return normalized_jones_in_a(pd_code) == {0: 1}

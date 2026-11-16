from collections.abc import Callable, Sequence

from numpy.random import Generator

from bin_packing.instance.instance import Instance

# Receiving the loads of the open bins where the item fits, as (bin index, load) pairs in opening order, and
# returning the chosen bin index.
type BinChooser = Callable[[list[tuple[int, int]], Generator], int]


def choose_first(candidates: list[tuple[int, int]], rng: Generator) -> int:
    """Returns: int: The earliest opened bin where the item fits."""
    return candidates[0][0]


def choose_best(candidates: list[tuple[int, int]], rng: Generator) -> int:
    """Returns: int: The fullest bin where the item fits; ties go to the earliest opened."""
    return max(candidates, key=lambda candidate: candidate[1])[0]


def choose_worst(candidates: list[tuple[int, int]], rng: Generator) -> int:
    """Returns: int: The emptiest bin where the item fits; ties go to the earliest opened."""
    return min(candidates, key=lambda candidate: candidate[1])[0]


def choose_random(candidates: list[tuple[int, int]], rng: Generator) -> int:
    """Returns: int: A bin drawn uniformly among those where the item fits."""
    return candidates[int(rng.integers(len(candidates)))][0]


BIN_SELECTION_RULES: dict[str, BinChooser] = {
    "first": choose_first,
    "best": choose_best,
    "worst": choose_worst,
    "random": choose_random,
}


def pack(order: Sequence[int], instance: Instance, rule: str, rng: Generator) -> list[list[int]]:
    """Packs items one by one in the given order, opening a new bin when the rule finds none where the item fits.

    Args:
        order (Sequence[int]): Item indices in the order they are packed.
        instance (Instance): Instance the items belong to.
        rule (str): Bin selection rule, one of BIN_SELECTION_RULES.
        rng (Generator): Source of randomness for the "random" rule.

    Returns:
        list[list[int]]: Item indices held by each bin, in opening order.

    Raises:
        ValueError: If the rule is unknown.
    """
    if rule not in BIN_SELECTION_RULES:
        raise ValueError(f"Invalid bin selection rule: {rule!r}. Expected one of {tuple(BIN_SELECTION_RULES)}.")
    choose = BIN_SELECTION_RULES[rule]
    sizes = instance.get_item_sizes()
    capacity = instance.get_capacity()
    bins: list[list[int]] = []
    loads: list[int] = []
    for item in order:
        size = sizes[item]
        candidates = [(index, load) for index, load in enumerate(loads) if load + size <= capacity]
        if candidates:
            chosen = choose(candidates, rng)
            bins[chosen].append(item)
            loads[chosen] += size
        else:
            bins.append([item])
            loads.append(size)
    return bins

import numpy as np

from bin_packing.instance.generator import generate_weibull_instance
from bin_packing.instance.instance import Instance
from bin_packing.instance.lower_bounds import lower_bound_l1, lower_bound_l2


def test_l1_rounds_total_size_up() -> None:
    assert lower_bound_l1(Instance("even", [6, 5, 4, 3, 2], 10)) == 2
    assert lower_bound_l1(Instance("odd", [6, 5, 4, 3, 3], 10)) == 3


def test_l2_counts_items_larger_than_half_capacity() -> None:
    # Three items above C/2 can never share a bin, while their total size alone gives L1 = 2.
    instance = Instance("halves", [6, 6, 6], 10)
    assert lower_bound_l1(instance) == 2
    assert lower_bound_l2(instance) == 3


def test_l2_counts_small_items_equal_to_alpha() -> None:
    # With alpha = 2, both 3s are large and the 2 is small; it cannot join either of them, so 3 bins are needed.
    assert lower_bound_l2(Instance("alpha", [3, 3, 2], 4)) == 3


def find_optimum(sizes: list[int], capacity: int) -> int:
    """Finds the minimum number of bins by exhaustive search; only usable for a handful of items.

    Args:
        sizes (list[int]): Item sizes.
        capacity (int): Capacity of each bin.

    Returns:
        int: The optimal number of bins.
    """
    best = len(sizes)

    def place(index: int, loads: list[int]) -> None:
        nonlocal best
        if len(loads) >= best:
            return
        if index == len(sizes):
            best = len(loads)
            return
        for bin_index, load in enumerate(loads):
            if load + sizes[index] <= capacity:
                loads[bin_index] += sizes[index]
                place(index + 1, loads)
                loads[bin_index] -= sizes[index]
        loads.append(sizes[index])
        place(index + 1, loads)
        loads.pop()

    place(0, [])
    return best


def test_l2_never_exceeds_the_optimum() -> None:
    rng = np.random.default_rng(0)
    for _ in range(500):
        capacity = int(rng.integers(5, 21))
        sizes = [int(size) for size in rng.integers(1, capacity + 1, size=rng.integers(1, 9))]
        assert lower_bound_l2(Instance("small", sizes, capacity)) <= find_optimum(sizes, capacity)


def test_l2_is_never_below_l1() -> None:
    for seed in range(10):
        instance = generate_weibull_instance(120, 150, "scaled", seed)
        assert lower_bound_l2(instance) >= lower_bound_l1(instance)

import math

from bin_packing.instance.instance import Instance


def lower_bound_l1(instance: Instance) -> int:
    """Computes the L1 lower bound: the total item size divided by the capacity, rounded up.

    Args:
        instance (Instance): Instance to bound.

    Returns:
        int: The minimum number of bins any feasible packing needs.
    """
    return math.ceil(sum(instance.get_item_sizes()) / instance.get_capacity())


def lower_bound_l2(instance: Instance) -> int:
    """Computes the L2 lower bound of Martello & Toth (1990).

    For each threshold alpha in [0, C/2], items split into large (above C - alpha), medium (above C/2 and up to
    C - alpha) and small (from alpha up to C/2). Large and medium items each need a bin of their own; small items
    can only fill the space the medium bins leave free, and any excess needs extra bins.

    Args:
        instance (Instance): Instance to bound.

    Returns:
        int: The minimum number of bins any feasible packing needs.
    """
    items = instance.get_item_sizes()
    capacity = instance.get_capacity()
    best = lower_bound_l1(instance)

    for alpha in range(capacity // 2 + 1):
        num_large = sum(1 for size in items if size > capacity - alpha)
        medium = [size for size in items if capacity - alpha >= size > capacity / 2]
        small_total = sum(size for size in items if capacity / 2 >= size >= alpha)

        free_space_in_medium_bins = sum(capacity - size for size in medium)
        overflow = max(0, small_total - free_space_in_medium_bins)
        best = max(best, num_large + len(medium) + math.ceil(overflow / capacity))
    return best

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

    Args:
        instance (Instance): Instance to bound.

    Returns:
        int: The minimum number of bins any feasible packing needs.
    """
    items = sorted(instance.get_item_sizes(), reverse=True)
    capacity = instance.get_capacity()
    best = lower_bound_l1(instance)

    for alpha in range(capacity // 2 + 1):
        big = [it for it in items if it > capacity - alpha]
        medium = [it for it in items if capacity - alpha >= it > capacity / 2]
        small_sum = sum(it for it in items if it <= capacity / 2 and it > alpha)

        l2 = len(big) + len(medium)
        used_cap = sum(capacity - it for it in medium) + len(big) * 0
        remaining = max(0, small_sum - used_cap)
        l2 += math.ceil(remaining / capacity)
        best = max(best, l2)
    return best

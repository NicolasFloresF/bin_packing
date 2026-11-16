import numpy as np


def generate_weibull_distribution(n_items: int, capacity: int, shape: float = 3.0, seed: int | None = None) -> dict:
    """Generates a bin packing instance whose item sizes follow a Weibull distribution.

    Args:
        n_items (int): Number of items to generate.
        capacity (int): Capacity of each bin.
        shape (float): Shape parameter of the Weibull distribution.
        seed (int | None): Random seed for reproducibility.

    Returns:
        dict: The generated 'bins', 'items' and an empty 'solution' list.
    """

    rng = np.random.default_rng(seed)

    # Scaling item sizes with the capacity so every capacity shares the same relative distribution and difficulty
    # stays comparable across capacities. The unscaled alternative is a fixed scale of 45.
    scale = capacity * (45 / 100)

    sizes = rng.weibull(shape, n_items) * scale
    sizes = np.round(sizes).astype(int)

    sizes = np.clip(sizes, 1, capacity)

    return {
        "bins": [capacity] * n_items,
        "items": sizes.tolist(),
        "solution": [],
    }


def lower_bound_l1(items: list[int], capacity: int) -> int:
    """Computes the L1 lower bound: the total item size divided by the capacity, rounded up.

    Args:
        items (list[int]): Item sizes.
        capacity (int): Capacity of each bin.

    Returns:
        int: The minimum number of bins any feasible packing needs.
    """
    return int(np.ceil(sum(items) / capacity))


def lower_bound_l2(items: list[int], capacity: int) -> int:
    """Computes the L2 lower bound of Martello & Toth (1990).

    Args:
        items (list[int]): Item sizes.
        capacity (int): Capacity of each bin.

    Returns:
        int: The minimum number of bins any feasible packing needs.
    """

    items = sorted(items, reverse=True)
    best = int(np.ceil(sum(items) / capacity))

    for alpha in range(capacity // 2 + 1):
        big = [it for it in items if it > capacity - alpha]
        medium = [it for it in items if capacity - alpha >= it > capacity / 2]
        small_sum = sum(it for it in items if it <= capacity / 2 and it > alpha)

        l2 = len(big) + len(medium)
        used_cap = sum(capacity - it for it in medium) + len(big) * 0
        remaining = max(0, small_sum - used_cap)
        l2 += int(np.ceil(remaining / capacity)) if capacity else 0
        best = max(best, l2)
    return best

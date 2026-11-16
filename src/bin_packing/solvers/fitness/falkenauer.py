from collections.abc import Sequence

# Squaring the fill ratios makes one full bin plus one nearly empty bin score higher than two half-full bins,
# which rewards packings that are close to freeing a bin.
FILL_EXPONENT = 2


def falkenauer_fitness(loads: Sequence[int], capacity: int) -> float:
    """Computes the Falkenauer fitness: the mean of the squared fill ratios of the used bins.

    Args:
        loads (Sequence[int]): Total item size of each bin; empty bins (load 0) are not counted as used.
        capacity (int): Capacity of each bin.

    Returns:
        float: The fitness; 1.0 when every used bin is exactly full, above 1.0 only for overloaded bins.
    """
    used_bins = sum(1 for load in loads if load > 0)
    return sum((load / capacity) ** FILL_EXPONENT for load in loads) / max(1, used_bins)

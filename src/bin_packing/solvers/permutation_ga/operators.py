from collections.abc import Callable

import numpy as np
from numpy.random import Generator

from bin_packing.instance.instance import Instance

TOURNAMENT_SIZE = 3


def create_random_population(instance: Instance, population_size: int, rng: Generator) -> list[list[int]]:
    """Returns: list[list[int]]: population_size item orders drawn uniformly at random."""
    num_items = instance.get_num_items()
    return [[int(item) for item in rng.permutation(num_items)] for _ in range(population_size)]


def create_ffd_seeded_population(instance: Instance, population_size: int, rng: Generator) -> list[list[int]]:
    """Returns: list[list[int]]: Random item orders, except the first, which is the decreasing-size order of FFD."""
    population = create_random_population(instance, population_size, rng)
    sizes = instance.get_item_sizes()
    population[0] = sorted(range(instance.get_num_items()), key=lambda item: -sizes[item])
    return population


def select_by_tournament(fitnesses: list[float], rng: Generator) -> int:
    """Returns: int: The fittest of TOURNAMENT_SIZE individuals drawn without replacement."""
    contenders = rng.choice(len(fitnesses), size=min(TOURNAMENT_SIZE, len(fitnesses)), replace=False)
    return int(max(contenders, key=lambda index: fitnesses[index]))


def cross_by_order(parent_a: list[int], parent_b: list[int], rng: Generator) -> list[int]:
    """Builds a child by the order crossover (OX): a slice of parent_a, the rest in parent_b's order.

    Args:
        parent_a (list[int]): Item order that gives the slice.
        parent_b (list[int]): Item order that fills the remaining positions, read from after the slice.
        rng (Generator): Source of randomness for the slice bounds.

    Returns:
        list[int]: A valid item order.
    """
    size = len(parent_a)
    start, end = sorted(int(position) for position in rng.choice(size, size=2, replace=False))
    kept = parent_a[start : end + 1]
    kept_items = set(kept)
    fill = [item for item in parent_b[end + 1 :] + parent_b[: end + 1] if item not in kept_items]
    # Placing the filling items after the slice first, wrapping around to the start.
    after_slice = size - end - 1
    return fill[after_slice:] + kept + fill[:after_slice]


def mutate_by_swap(order: list[int], rate: float, rng: Generator) -> list[int]:
    """Swaps each position, with probability rate, with a position drawn at random.

    Args:
        order (list[int]): Item order; left unchanged.
        rate (float): Probability of mutating each position.
        rng (Generator): Source of randomness.

    Returns:
        list[int]: The mutated item order.
    """
    mutated = list(order)
    positions = np.flatnonzero(rng.random(len(order)) < rate)
    for position, other in zip(positions, rng.integers(len(order), size=len(positions)), strict=True):
        mutated[position], mutated[other] = mutated[other], mutated[position]
    return mutated


INITIALIZATIONS: dict[str, Callable[[Instance, int, Generator], list[list[int]]]] = {
    "random": create_random_population,
    "ffd_seeded": create_ffd_seeded_population,
}
SELECTIONS: dict[str, Callable[[list[float], Generator], int]] = {
    "tournament": select_by_tournament,
}
CROSSOVERS: dict[str, Callable[[list[int], list[int], Generator], list[int]]] = {
    "order": cross_by_order,
}
MUTATIONS: dict[str, Callable[[list[int], float, Generator], list[int]]] = {
    "swap": mutate_by_swap,
}

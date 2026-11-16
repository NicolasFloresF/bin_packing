from collections.abc import Callable

import numpy as np
from numpy.random import Generator

from bin_packing.instance.instance import Instance
from bin_packing.solvers.fitness.falkenauer import falkenauer_fitness

# Giving up after this many draws and keeping the last bin drawn, as the graduation GA did.
PLACEMENT_TRIES = 100
UNIFORM_PARENT_PROBABILITY = 0.5


def create_random_individual(instance: Instance, bin_limit: int, rng: Generator) -> list[int]:
    """Creates a chromosome by drawing, for each item, random bins until one fits or the tries run out.

    Args:
        instance (Instance): Instance being solved.
        bin_limit (int): Number of bins the chromosome may use.
        rng (Generator): Source of randomness.

    Returns:
        list[int]: The bin index of each item; bins may end up overloaded.
    """
    loads = [0] * bin_limit
    chromosome = []
    capacity = instance.get_capacity()
    for size in instance.get_item_sizes():
        draws = rng.integers(bin_limit, size=PLACEMENT_TRIES)
        fitting = [int(bin_index) for bin_index in draws if loads[bin_index] + size <= capacity]
        chosen = fitting[0] if fitting else int(draws[-1])
        chromosome.append(chosen)
        loads[chosen] += size
    return chromosome


def compute_loads(chromosome: list[int], instance: Instance, bin_limit: int) -> list[int]:
    """Returns: list[int]: Total item size assigned to each of the bin_limit bins."""
    loads = [0] * bin_limit
    for size, bin_index in zip(instance.get_item_sizes(), chromosome, strict=True):
        loads[bin_index] += size
    return loads


def compute_penalized_fitness(loads: list[int], capacity: int, penalty_severity: float) -> float:
    """Computes the Falkenauer fitness capped at 1, divided by the overload penalty raised to the severity.

    Args:
        loads (list[int]): Total item size of each bin.
        capacity (int): Capacity of each bin.
        penalty_severity (float): Exponent of the penalty; 1 reproduces the graduation GA.

    Returns:
        float: The fitness; feasible packings are not penalized.
    """
    overload = sum(load / capacity for load in loads if load > capacity)
    return min(1.0, falkenauer_fitness(loads, capacity)) / max(1.0, overload) ** penalty_severity


def select_by_roulette_wheel(fitnesses: list[float], count: int, rng: Generator) -> list[int]:
    """Draws individuals with probability proportional to their fitness, with replacement.

    Args:
        fitnesses (list[float]): Fitness of each individual of the population.
        count (int): Number of individuals to draw.
        rng (Generator): Source of randomness.

    Returns:
        list[int]: Indices of the drawn individuals.
    """
    probabilities = np.asarray(fitnesses, dtype=float)
    probabilities /= probabilities.sum()
    return [int(index) for index in rng.choice(len(fitnesses), size=count, p=probabilities)]


def cross_at_one_point(parent_a: list[int], parent_b: list[int], rng: Generator) -> list[int]:
    """Returns: list[int]: The first half of parent_a followed by the second half of parent_b."""
    middle = (len(parent_a) + 1) // 2
    return parent_a[:middle] + parent_b[middle:]


def cross_uniformly(parent_a: list[int], parent_b: list[int], rng: Generator) -> list[int]:
    """Returns: list[int]: Each gene taken from parent_a or parent_b with equal probability."""
    from_a = rng.random(len(parent_a)) < UNIFORM_PARENT_PROBABILITY
    return [gene_a if take_a else gene_b for gene_a, gene_b, take_a in zip(parent_a, parent_b, from_a, strict=True)]


def mutate_by_reassignment(chromosome: list[int], bin_limit: int, rate: float, rng: Generator) -> list[int]:
    """Moves each item, with probability rate, to a bin drawn at random (possibly its current one).

    Args:
        chromosome (list[int]): Bin index of each item; left unchanged.
        bin_limit (int): Number of bins the chromosome may use.
        rate (float): Probability of mutating each gene.
        rng (Generator): Source of randomness.

    Returns:
        list[int]: The mutated chromosome.
    """
    mutated = list(chromosome)
    positions = np.flatnonzero(rng.random(len(chromosome)) < rate)
    for position, bin_index in zip(positions, rng.integers(bin_limit, size=len(positions)), strict=True):
        mutated[position] = int(bin_index)
    return mutated


SELECTIONS: dict[str, Callable[[list[float], int, Generator], list[int]]] = {
    "roulette_wheel": select_by_roulette_wheel,
}
CROSSOVERS: dict[str, Callable[[list[int], list[int], Generator], list[int]]] = {
    "one_point": cross_at_one_point,
    "uniform": cross_uniformly,
}
MUTATIONS: dict[str, Callable[[list[int], int, float, Generator], list[int]]] = {
    "reassign": mutate_by_reassignment,
}

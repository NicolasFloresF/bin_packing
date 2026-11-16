import numpy as np

from bin_packing.instance.instance import Instance
from bin_packing.solvers.permutation_ga.operators import (
    create_ffd_seeded_population,
    create_random_population,
    cross_by_order,
    mutate_by_swap,
    select_by_tournament,
)

INSTANCE = Instance("toy", [4, 6, 2, 5, 3], 10)


def test_random_population_holds_permutations() -> None:
    population = create_random_population(INSTANCE, 10, np.random.default_rng(0))
    assert len(population) == 10
    assert all(sorted(order) == [0, 1, 2, 3, 4] for order in population)


def test_ffd_seeded_population_starts_with_the_decreasing_order() -> None:
    population = create_ffd_seeded_population(INSTANCE, 10, np.random.default_rng(0))
    assert population[0] == [1, 3, 0, 4, 2]
    assert len(population) == 10


def test_order_crossover_keeps_a_slice_of_the_first_parent_and_yields_permutations() -> None:
    parent_a = [1, 2, 0, 3, 4, 7, 5, 6]
    parent_b = [7, 6, 5, 4, 3, 2, 1, 0]
    for seed in range(50):
        child = cross_by_order(parent_a, parent_b, np.random.default_rng(seed))
        assert sorted(child) == list(range(8))
        kept = [position for position in range(8) if child[position] == parent_a[position]]
        assert len(kept) >= 2


def test_order_crossover_of_identical_parents_returns_the_parent() -> None:
    parent = [3, 1, 4, 0, 2]
    assert cross_by_order(parent, parent, np.random.default_rng(0)) == parent


def test_swap_mutation_preserves_the_permutation() -> None:
    rng = np.random.default_rng(0)
    order = list(range(100))
    assert mutate_by_swap(order, 0.0, rng) == order
    mutated = mutate_by_swap(order, 0.5, rng)
    assert sorted(mutated) == order
    assert mutated != order


def test_tournament_returns_the_fittest_contender() -> None:
    assert select_by_tournament([0.1, 0.9, 0.5], np.random.default_rng(0)) == 1

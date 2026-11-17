import numpy as np
import pytest

from bin_packing.instance.generator import generate_weibull_instance
from bin_packing.instance.instance import Instance
from bin_packing.solvers.heuristics.decreasing_fit import DecreasingFitHeuristic
from bin_packing.solvers.permutation_ga.genetic_algorithm import PermutationGeneticAlgorithm

INSTANCE = generate_weibull_instance(40, 150, "scaled", 0)


def test_run_returns_a_feasible_reproducible_solution_with_history() -> None:
    genetic_algorithm = PermutationGeneticAlgorithm("PGA", {"generations": 50})
    first = genetic_algorithm.solve(INSTANCE, np.random.default_rng(3))
    history = genetic_algorithm.get_history()
    second = genetic_algorithm.solve(INSTANCE, np.random.default_rng(3))
    assert first.is_feasible()
    assert first.get_bins() == second.get_bins()
    assert 0 < len(history) <= 50
    assert all(best >= median >= worst for best, median, worst in history)


def test_ffd_seeded_run_is_never_worse_than_ffd() -> None:
    rng = np.random.default_rng(0)
    for seed in range(3):
        instance = generate_weibull_instance(60, 150, "scaled", seed)
        ffd_bins = DecreasingFitHeuristic("FFD", "first").solve(instance, rng).get_num_bins()
        genetic_algorithm = PermutationGeneticAlgorithm("PGA", {"generations": 5, "initialization": "ffd_seeded"})
        assert genetic_algorithm.solve(instance, rng).get_num_bins() <= ffd_bins


def test_run_stops_early_when_reaching_the_lower_bound() -> None:
    genetic_algorithm = PermutationGeneticAlgorithm("PGA", {"generations": 1000})
    solution = genetic_algorithm.solve(Instance("pairs", [5] * 8, 10), np.random.default_rng(0))
    assert solution.get_num_bins() == 4
    assert genetic_algorithm.get_history() == []


@pytest.mark.parametrize(
    ("arguments", "message"),
    [({"initialization": "greedy"}, "initialization"), ({"population_size": 0}, "population_size")],
)
def test_invalid_arguments_are_rejected(arguments: dict[str, int | str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PermutationGeneticAlgorithm("PGA", arguments)

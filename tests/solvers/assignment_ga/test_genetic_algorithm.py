import numpy as np
import pytest

from bin_packing.instance.generator import generate_weibull_instance
from bin_packing.instance.instance import Instance
from bin_packing.instance.lower_bounds import lower_bound_l2
from bin_packing.solvers.assignment_ga.genetic_algorithm import AssignmentGeneticAlgorithm, Individual
from bin_packing.solvers.solution import Solution

INSTANCE = generate_weibull_instance(40, 150, "scaled", 0)


def test_run_returns_a_feasible_reproducible_solution_with_history() -> None:
    genetic_algorithm = AssignmentGeneticAlgorithm("AGA", {"generations": 100})
    first = genetic_algorithm.solve(INSTANCE, np.random.default_rng(3))
    history = genetic_algorithm.get_history()
    second = genetic_algorithm.solve(INSTANCE, np.random.default_rng(3))
    assert isinstance(first, Solution)
    assert first.is_feasible()
    assert first.get_bins() == second.get_bins()
    assert 0 < len(history) <= 100
    assert all(best >= median >= worst for best, median, worst in history)


def test_run_stops_early_when_reaching_the_lower_bound() -> None:
    instance = Instance("pairs", [5] * 8, 10)
    genetic_algorithm = AssignmentGeneticAlgorithm("AGA", {"generations": 1000})
    solution = genetic_algorithm.solve(instance, np.random.default_rng(0))
    assert solution.get_num_bins() == lower_bound_l2(instance) == 4
    assert len(genetic_algorithm.get_history()) < 1000


def test_original_operators_are_available_by_name() -> None:
    original = {
        "generations": 20,
        "crossover": "uniform",
        "mutation_rate": 0.2,
        "penalty_severity": 1,
        "offspring_per_generation": 25,
    }
    genetic_algorithm = AssignmentGeneticAlgorithm("AGA (original)", original)
    solution = genetic_algorithm.solve(INSTANCE, np.random.default_rng(0))
    assert solution.get_instance() is INSTANCE


def test_result_falls_back_to_the_fittest_infeasible_individual() -> None:
    genetic_algorithm = AssignmentGeneticAlgorithm("AGA", {})
    overloaded = Individual([0, 0], [12, 0], 0.5)
    also_overloaded = Individual([0, 1], [11, 1], 0.4)
    assert genetic_algorithm.pick_result([overloaded, also_overloaded], 10) is overloaded
    feasible = Individual([0, 1], [6, 6], 0.3)
    assert genetic_algorithm.pick_result([overloaded, feasible], 10) is feasible


@pytest.mark.parametrize(
    ("arguments", "message"),
    [({"crossover": "two_point"}, "crossover"), ({"mutation_rate": 2}, "mutation_rate"), ({"size": 5}, "size")],
)
def test_invalid_arguments_are_rejected(arguments: dict[str, int | str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AssignmentGeneticAlgorithm("AGA", arguments)

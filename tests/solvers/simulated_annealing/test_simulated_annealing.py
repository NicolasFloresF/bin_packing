import numpy as np
import pytest

from bin_packing.instance.generator import generate_weibull_instance
from bin_packing.instance.instance import Instance
from bin_packing.solvers.fitness.falkenauer import falkenauer_fitness
from bin_packing.solvers.heuristics.decreasing_fit import DecreasingFitHeuristic
from bin_packing.solvers.simulated_annealing.simulated_annealing import SimulatedAnnealing

INSTANCE = generate_weibull_instance(60, 150, "scaled", 0)
SHORT_SCHEDULE = {"iterations_per_temperature": 100}


def test_default_schedule_has_135_levels() -> None:
    # 1e-5 * 0.95**134 = 1.03e-8 is the last temperature not below 1e-8.
    annealing = SimulatedAnnealing("SA", {})
    assert annealing.get_num_levels() == 135


def test_run_returns_a_feasible_reproducible_solution() -> None:
    annealing = SimulatedAnnealing("SA", SHORT_SCHEDULE)
    first = annealing.solve(INSTANCE, np.random.default_rng(4))
    second = annealing.solve(INSTANCE, np.random.default_rng(4))
    assert first.is_feasible()
    assert first.get_bins() == second.get_bins()


def test_ffd_seeded_run_is_never_less_fit_than_ffd() -> None:
    rng = np.random.default_rng(0)
    for seed in range(3):
        instance = generate_weibull_instance(60, 150, "scaled", seed)
        ffd = DecreasingFitHeuristic("FFD", "first").solve(instance, rng)
        annealing = SimulatedAnnealing("SA", {**SHORT_SCHEDULE, "initialization": "ffd_seeded"})
        annealed = annealing.solve(instance, rng)
        assert falkenauer_fitness(annealed.get_bin_loads(), 150) >= falkenauer_fitness(ffd.get_bin_loads(), 150)


def test_run_stops_when_reaching_the_lower_bound() -> None:
    instance = Instance("pairs", [5] * 8, 10)
    for initialization in ("random", "ffd_seeded"):
        annealing = SimulatedAnnealing("SA", {"initialization": initialization})
        assert annealing.solve(instance, np.random.default_rng(0)).get_num_bins() == 4


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"cooling_rate": 1.0}, "cooling_rate"),
        ({"cooling_rate": 0.0}, "cooling_rate"),
        ({"final_temperature": 1e-4}, "final_temperature"),
        ({"initialization": "greedy"}, "initialization"),
        ({"iterations": 1}, "iterations"),
    ],
)
def test_invalid_arguments_are_rejected(arguments: dict[str, float | str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SimulatedAnnealing("SA", arguments)

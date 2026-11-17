import numpy as np
import pytest

from bin_packing.instance.generator import generate_weibull_instance
from bin_packing.instance.instance import Instance
from bin_packing.solvers.heuristics.bin_selection import pack
from bin_packing.solvers.heuristics.decreasing_fit import DecreasingFitHeuristic


def sizes_per_bin(rule: str, instance: Instance) -> list[list[int]]:
    """Returns: list[list[int]]: Item sizes per bin of the decreasing-order packing under one rule."""
    solution = DecreasingFitHeuristic(rule, rule).solve(instance, np.random.default_rng(0))
    sizes = instance.get_item_sizes()
    return [[sizes[item] for item in items] for items in solution.get_bins()]


# Packing 5, 3, 3, 3, 1 with C = 7: the 1 goes to the earliest bin (First), the fullest (Best) or the emptiest (Worst).
TOY = Instance("toy", [3, 1, 5, 3, 3], 7)


def test_first_fit_decreasing_uses_the_earliest_bin() -> None:
    assert sizes_per_bin("first", TOY) == [[5, 1], [3, 3], [3]]


def test_best_fit_decreasing_uses_the_fullest_bin() -> None:
    assert sizes_per_bin("best", TOY) == [[5], [3, 3, 1], [3]]


def test_worst_fit_decreasing_uses_the_emptiest_bin() -> None:
    assert sizes_per_bin("worst", TOY) == [[5], [3, 3], [3, 1]]


@pytest.mark.parametrize("rule", ["first", "best", "worst", "random"])
def test_every_rule_returns_a_feasible_packing(rule: str) -> None:
    instance = generate_weibull_instance(120, 150, "scaled", 0)
    solution = DecreasingFitHeuristic(rule, rule).solve(instance, np.random.default_rng(1))
    assert solution.is_feasible()
    assert solution.get_num_bins() >= 1


def test_random_fit_is_reproducible_by_seed_and_varies_across_seeds() -> None:
    instance = generate_weibull_instance(120, 150, "scaled", 0)
    heuristic = DecreasingFitHeuristic("RFD", "random")
    first = heuristic.solve(instance, np.random.default_rng(5)).get_bins()
    assert first == heuristic.solve(instance, np.random.default_rng(5)).get_bins()
    assert first != heuristic.solve(instance, np.random.default_rng(6)).get_bins()


def test_unknown_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="'almost'"):
        DecreasingFitHeuristic("AFD", "almost")
    with pytest.raises(ValueError, match="'almost'"):
        pack([0], TOY, "almost", np.random.default_rng(0))

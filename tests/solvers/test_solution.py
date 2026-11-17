import pytest

from bin_packing.instance.instance import Instance
from bin_packing.solvers.heuristics.decreasing_fit import DecreasingFitHeuristic
from bin_packing.solvers.solution import Solution

INSTANCE = Instance("toy", [6, 5, 4, 3, 2], 10)


def test_feasible_solution_reports_loads_and_bins() -> None:
    solution = Solution(INSTANCE, [[0, 2], [1, 3, 4]])
    assert solution.get_instance() is INSTANCE
    assert solution.get_num_bins() == 2
    assert solution.get_bin_loads() == [10, 10]
    assert solution.is_feasible()


def test_overloaded_solution_is_kept_but_infeasible() -> None:
    solution = Solution(INSTANCE, [[0, 1], [2, 3, 4]])
    assert solution.get_bin_loads() == [11, 9]
    assert not solution.is_feasible()


def test_returned_bins_are_a_copy() -> None:
    solution = Solution(INSTANCE, [[0, 2], [1, 3, 4]])
    solution.get_bins()[0].append(1)
    assert solution.get_bins() == [[0, 2], [1, 3, 4]]


@pytest.mark.parametrize(
    ("bins", "message"),
    [
        ([[0, 2], [], [1, 3, 4]], "Bin 1 .* is empty"),
        ([[0, 2, 7], [1, 3, 4]], "item index 7"),
        ([[0, 2], [1, 3, 4, 2]], "Item 2 appears more than once"),
        ([[0, 2], [1, 3]], r"Items \[4\] are missing"),
    ],
)
def test_invalid_packings_are_rejected(bins: list[list[int]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Solution(INSTANCE, bins)


def test_solver_name_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="solver name"):
        DecreasingFitHeuristic("", "first")
    assert DecreasingFitHeuristic("FFD", "first").get_name() == "FFD"

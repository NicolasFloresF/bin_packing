import pytest

from bin_packing.solvers.fitness.falkenauer import falkenauer_fitness


def test_full_bins_score_one_and_empty_bins_are_not_counted() -> None:
    assert falkenauer_fitness([10, 10], 10) == 1.0
    assert falkenauer_fitness([10, 0, 10], 10) == 1.0


def test_nearly_free_bin_scores_higher_than_evenly_filled_bins() -> None:
    nearly_free = falkenauer_fitness([9, 9, 2], 10)
    evenly_filled = falkenauer_fitness([7, 7, 6], 10)
    assert nearly_free == pytest.approx(0.553, abs=1e-3)
    assert evenly_filled == pytest.approx(0.447, abs=1e-3)
    assert nearly_free > evenly_filled

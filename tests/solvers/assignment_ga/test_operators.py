import numpy as np
import pytest

from bin_packing.instance.instance import Instance
from bin_packing.solvers.assignment_ga.operators import (
    compute_loads,
    compute_penalized_fitness,
    create_random_individual,
    cross_at_one_point,
    cross_uniformly,
    mutate_by_reassignment,
    select_by_roulette_wheel,
)

INSTANCE = Instance("toy", [6, 5, 4, 3, 2], 10)


def test_random_individual_fits_items_when_bins_are_plentiful() -> None:
    chromosome = create_random_individual(INSTANCE, 5, np.random.default_rng(0))
    loads = compute_loads(chromosome, INSTANCE, 5)
    assert all(0 <= bin_index < 5 for bin_index in chromosome)
    assert sum(loads) == 20
    assert max(loads) <= 10


def test_random_individual_overloads_when_bins_are_too_few() -> None:
    chromosome = create_random_individual(INSTANCE, 1, np.random.default_rng(0))
    assert chromosome == [0, 0, 0, 0, 0]


def test_penalty_severity_one_reproduces_the_lenient_original_penalty() -> None:
    # The overloaded (11, 9) packing outscores the feasible (10, 7, 3) one unless the penalty is harsher.
    assert compute_penalized_fitness([11, 9], 10, 1.0) == pytest.approx(0.909, abs=1e-3)
    assert compute_penalized_fitness([10, 7, 3], 10, 1.0) == pytest.approx(0.527, abs=1e-3)
    assert compute_penalized_fitness([11, 9], 10, 10.0) == pytest.approx(0.386, abs=1e-3)
    assert compute_penalized_fitness([10, 7, 3], 10, 10.0) == pytest.approx(0.527, abs=1e-3)


def test_one_point_crossover_splits_at_the_middle() -> None:
    rng = np.random.default_rng(0)
    assert cross_at_one_point([0, 0, 0, 0, 0], [1, 1, 1, 1, 1], rng) == [0, 0, 0, 1, 1]
    assert cross_at_one_point([0, 0, 0, 0], [1, 1, 1, 1], rng) == [0, 0, 1, 1]


def test_uniform_crossover_takes_each_gene_from_one_parent() -> None:
    child = cross_uniformly([0] * 200, [1] * 200, np.random.default_rng(0))
    assert len(child) == 200
    assert 50 < sum(child) < 150


def test_reassignment_respects_rate_and_bin_limit() -> None:
    rng = np.random.default_rng(0)
    chromosome = [0, 1, 2, 3, 4]
    assert mutate_by_reassignment(chromosome, 7, 0.0, rng) == chromosome
    mutated = mutate_by_reassignment([0] * 500, 7, 1.0, rng)
    assert set(mutated) == set(range(7))


def test_roulette_wheel_favors_fitter_individuals() -> None:
    drawn = select_by_roulette_wheel([0.9, 0.1], 1000, np.random.default_rng(0))
    assert 800 < drawn.count(0) < 980

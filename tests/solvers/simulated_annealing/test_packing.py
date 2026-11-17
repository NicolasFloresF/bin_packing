import numpy as np
import pytest

from bin_packing.instance.generator import generate_weibull_instance
from bin_packing.instance.instance import Instance
from bin_packing.solvers.fitness.falkenauer import falkenauer_fitness
from bin_packing.solvers.heuristics.bin_selection import pack
from bin_packing.solvers.simulated_annealing.packing import Packing

INSTANCE = Instance("toy", [6, 5, 4, 3, 2], 10)


def loads_of(packing: Packing, instance: Instance) -> list[int]:
    """Returns: list[int]: Load of each used bin, recomputed from scratch."""
    sizes = instance.get_item_sizes()
    return [sum(sizes[item] for item in items) for items in packing.get_used_bins()]


def test_initial_fitness_matches_falkenauer() -> None:
    packing = Packing(INSTANCE, [[0, 4], [1, 3], [2]])
    assert packing.get_fitness() == pytest.approx(falkenauer_fitness([8, 8, 4], 10))
    assert packing.get_num_used_bins() == 3


def test_overloaded_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="above the capacity"):
        Packing(INSTANCE, [[0, 1], [2, 3, 4]])


def test_infeasible_or_pointless_steps_are_refused() -> None:
    packing = Packing(INSTANCE, [[0, 4], [1, 3], [2]])
    assert packing.fitness_after_move(0, 1) is None  # 6 does not fit next to 5 + 3
    assert packing.fitness_after_move(0, 0) is None  # same bin
    assert packing.fitness_after_swap(0, 4) is None  # same bin
    assert packing.fitness_after_swap(0, 3) is None  # 3 + 4 would leave 5 + 6 = 11 in the other bin


def test_emptying_a_bin_frees_it() -> None:
    # Loads 10, 8, 2: moving the 2 into the bin holding 5 + 3 fills it and empties the third bin.
    packing = Packing(INSTANCE, [[0, 2], [1, 3], [4]])
    predicted = packing.fitness_after_move(4, 1)
    packing.apply_move(4, 1)
    assert packing.get_num_used_bins() == 2
    assert packing.get_fitness() == pytest.approx(1.0) == predicted
    assert packing.fitness_after_move(0, 2) is None  # the emptied bin does not reopen


def test_predicted_fitness_matches_the_applied_step() -> None:
    instance = generate_weibull_instance(60, 150, "scaled", 0)
    rng = np.random.default_rng(0)
    packing = Packing(instance, pack(list(range(60)), instance, "first", rng))
    applied = 0
    for _ in range(2000):
        item, other = int(rng.integers(60)), int(rng.integers(60))
        target = int(rng.integers(packing.get_num_bins()))
        swap = bool(rng.random() < 0.5)
        predicted = packing.fitness_after_swap(item, other) if swap else packing.fitness_after_move(item, target)
        if predicted is None:
            continue
        packing.apply_swap(item, other) if swap else packing.apply_move(item, target)
        applied += 1
        assert packing.get_fitness() == pytest.approx(predicted)
        assert packing.get_fitness() == pytest.approx(falkenauer_fitness(loads_of(packing, instance), 150))
        assert max(loads_of(packing, instance)) <= 150
        for index, items in enumerate(packing.bins):
            assert all(packing.get_bin_of_item(member) == index for member in items)
    assert applied > 100

from bin_packing.instance.generator import generate_weibull_instance
from bin_packing.instance.instance import Instance
from bin_packing.instance.lower_bounds import lower_bound_l1, lower_bound_l2


def test_l1_rounds_total_size_up() -> None:
    assert lower_bound_l1(Instance("even", [6, 5, 4, 3, 2], 10)) == 2
    assert lower_bound_l1(Instance("odd", [6, 5, 4, 3, 3], 10)) == 3


def test_l2_counts_items_larger_than_half_capacity() -> None:
    # Three items above C/2 can never share a bin, while their total size alone gives L1 = 2.
    instance = Instance("halves", [6, 6, 6], 10)
    assert lower_bound_l1(instance) == 2
    assert lower_bound_l2(instance) == 3


def test_l2_is_never_below_l1() -> None:
    for seed in range(10):
        instance = generate_weibull_instance(120, 150, "scaled", seed)
        assert lower_bound_l2(instance) >= lower_bound_l1(instance)

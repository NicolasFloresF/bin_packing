from bin_packing.instance.helper_functions import generate_weibull_distribution, lower_bound_l1, lower_bound_l2


def test_l1_rounds_total_size_up() -> None:
    assert lower_bound_l1([6, 5, 4, 3, 2], 10) == 2
    assert lower_bound_l1([6, 5, 4, 3, 3], 10) == 3


def test_l2_counts_items_larger_than_half_capacity() -> None:
    # Three items above C/2 can never share a bin, while their total size alone gives L1 = 2.
    assert lower_bound_l1([6, 6, 6], 10) == 2
    assert lower_bound_l2([6, 6, 6], 10) == 3


def test_l2_is_never_below_l1() -> None:
    items = generate_weibull_distribution(120, 150, seed=3)["items"]
    assert lower_bound_l2(items, 150) >= lower_bound_l1(items, 150)


def test_generator_is_reproducible_and_within_capacity() -> None:
    first = generate_weibull_distribution(200, 50, seed=7)
    second = generate_weibull_distribution(200, 50, seed=7)
    assert first["items"] == second["items"]
    assert len(first["items"]) == 200
    assert all(1 <= size <= 50 for size in first["items"])

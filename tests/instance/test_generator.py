import pytest

from bin_packing.instance.generator import generate_weibull_instance


def test_generator_is_reproducible_and_named_after_its_parameters() -> None:
    first = generate_weibull_instance(200, 150, "scaled", 7)
    second = generate_weibull_instance(200, 150, "scaled", 7)
    assert first.get_item_sizes() == second.get_item_sizes()
    assert first.get_name() == "weibull_scaled_n200_c150_s7"
    assert first.get_num_items() == 200


def test_scaled_sizes_grow_with_capacity_and_unscaled_do_not() -> None:
    small = generate_weibull_instance(500, 150, "scaled", 0)
    large = generate_weibull_instance(500, 1500, "scaled", 0)
    assert sum(large.get_item_sizes()) > 9 * sum(small.get_item_sizes())
    small = generate_weibull_instance(500, 150, "unscaled", 0)
    large = generate_weibull_instance(500, 1500, "unscaled", 0)
    assert small.get_item_sizes() == large.get_item_sizes()


def test_generator_warns_when_many_sizes_are_clipped() -> None:
    with pytest.warns(UserWarning, match="clipped to \\[1, 50\\]"):
        instance = generate_weibull_instance(500, 50, "unscaled", 0)
    assert max(instance.get_item_sizes()) == 50


@pytest.mark.parametrize(
    ("num_items", "size_scaling", "message"),
    [(10, "linear", "size scaling"), (0, "scaled", "items")],
)
def test_generator_rejects_invalid_arguments(num_items: int, size_scaling: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_weibull_instance(num_items, 150, size_scaling, 0)

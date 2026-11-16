import pytest

from bin_packing.instance.instance import Instance


def test_instance_exposes_its_data() -> None:
    instance = Instance("toy", [6, 5, 4], 10)
    assert instance.get_name() == "toy"
    assert instance.get_item_sizes() == (6, 5, 4)
    assert instance.get_capacity() == 10
    assert instance.get_num_items() == 3


@pytest.mark.parametrize(
    ("name", "sizes", "capacity", "message"),
    [
        ("", [1], 10, "instance name"),
        ("toy", [1], 0, "capacity"),
        ("toy", [], 10, "no items"),
        ("toy", [4, 11], 10, "size 11 for item 1"),
        ("toy", [0], 10, "size 0 for item 0"),
    ],
)
def test_instance_rejects_invalid_data(name: str, sizes: list[int], capacity: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Instance(name, sizes, capacity)

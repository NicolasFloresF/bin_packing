import pytest

from bin_packing.solvers.arguments import (
    Arguments,
    merge_arguments,
    require_choice,
    require_number,
    require_optional_number,
    require_positive_int,
)

DEFAULTS: Arguments = {"generations": 10, "crossover": "one_point", "rate": None}


def test_given_arguments_override_defaults() -> None:
    assert merge_arguments("GA", {"generations": 3}, DEFAULTS) == {
        "generations": 3,
        "crossover": "one_point",
        "rate": None,
    }


def test_unknown_argument_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match=r"Unknown arguments \['generation'\] for solver 'GA'"):
        merge_arguments("GA", {"generation": 3}, DEFAULTS)


@pytest.mark.parametrize("value", [0, -1, 2.5, "3", True])
def test_positive_int_rejects_other_values(value: int | float | str) -> None:
    with pytest.raises(ValueError, match="generations"):
        require_positive_int({"generations": value}, "generations")


def test_choice_number_and_optional_number_checks() -> None:
    assert require_choice({"crossover": "uniform"}, "crossover", {"uniform"}) == "uniform"
    with pytest.raises(ValueError, match="'two_point'"):
        require_choice({"crossover": "two_point"}, "crossover", {"uniform"})
    assert require_number({"rate": 1}, "rate", 0.0, 1.0) == 1.0
    with pytest.raises(ValueError, match="rate"):
        require_number({"rate": 1.5}, "rate", 0.0, 1.0)
    assert require_optional_number({"rate": None}, "rate", 0.0, 1.0) is None
    assert require_optional_number({"rate": 0.2}, "rate", 0.0, 1.0) == 0.2

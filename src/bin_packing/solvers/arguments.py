from collections.abc import Collection, Mapping

type ArgumentValue = int | float | str | None
type Arguments = Mapping[str, ArgumentValue]


def merge_arguments(solver_name: str, given: Arguments, defaults: Arguments) -> dict[str, ArgumentValue]:
    """Returns the defaults overridden by the given arguments, after rejecting names the solver does not know.

    Args:
        solver_name (str): Name of the solver configuration, used in error messages.
        given (Arguments): Arguments supplied by the caller.
        defaults (Arguments): Every argument the solver accepts, with its default value.

    Returns:
        dict[str, ArgumentValue]: The complete set of arguments.

    Raises:
        ValueError: If an argument name is unknown.
    """
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        known = sorted(defaults)
        raise ValueError(f"Unknown arguments {unknown} for solver {solver_name!r}. Expected some of {known}.")
    return {**defaults, **given}


def require_positive_int(arguments: Arguments, key: str) -> int:
    """Returns: int: The argument, after checking it is an integer of at least 1 (raises ValueError otherwise)."""
    value = arguments[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Invalid {key}: {value!r}. Expected a positive integer.")
    return value


def require_choice(arguments: Arguments, key: str, choices: Collection[str]) -> str:
    """Returns: str: The argument, after checking it is one of the choices (raises ValueError otherwise)."""
    value = arguments[key]
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"Invalid {key}: {value!r}. Expected one of {sorted(choices)}.")
    return value


def require_number(arguments: Arguments, key: str, minimum: float, maximum: float) -> float:
    """Returns: float: The argument, after checking it is a number within [minimum, maximum] (raises ValueError)."""
    value = arguments[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {key}: {value!r}. Expected a number from {minimum} to {maximum}.")
    return float(value)


def require_optional_number(arguments: Arguments, key: str, minimum: float, maximum: float) -> float | None:
    """Returns: float | None: The argument when given, checked as require_number does; None when it is None."""
    if arguments[key] is None:
        return None
    return require_number(arguments, key, minimum, maximum)

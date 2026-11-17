import tomllib
from collections.abc import Callable
from pathlib import Path

from bin_packing.solvers.arguments import Arguments, ArgumentValue, merge_arguments, require_choice
from bin_packing.solvers.assignment_ga.genetic_algorithm import AssignmentGeneticAlgorithm
from bin_packing.solvers.heuristics.bin_selection import BIN_SELECTION_RULES
from bin_packing.solvers.heuristics.decreasing_fit import DecreasingFitHeuristic
from bin_packing.solvers.permutation_ga.genetic_algorithm import PermutationGeneticAlgorithm
from bin_packing.solvers.simulated_annealing.simulated_annealing import SimulatedAnnealing
from bin_packing.solvers.solver import Solver

EXPERIMENT_KEYS = ("instances", "seeds", "out")


def create_heuristic(name: str, arguments: Arguments) -> DecreasingFitHeuristic:
    """Builds a decreasing-fit heuristic from named arguments; bin_selection_rule defaults to "first".

    Args:
        name (str): Unique name of the configuration.
        arguments (Arguments): Optionally bin_selection_rule, one of "first", "best", "worst" or "random".

    Returns:
        DecreasingFitHeuristic: The configured heuristic.

    Raises:
        ValueError: If an argument is unknown or the rule is invalid.
    """
    merged = merge_arguments(name, arguments, {"bin_selection_rule": "first"})
    return DecreasingFitHeuristic(name, require_choice(merged, "bin_selection_rule", BIN_SELECTION_RULES))


SOLVER_TYPES: dict[str, Callable[[str, Arguments], Solver]] = {
    "heuristic": create_heuristic,
    "assignment_ga": AssignmentGeneticAlgorithm,
    "permutation_ga": PermutationGeneticAlgorithm,
    "simulated_annealing": SimulatedAnnealing,
}


class SolverConfiguration:
    """Holds what is needed to build one solver configuration, in a form that can be sent to worker processes."""

    def __init__(self, name: str, solver_type: str, arguments: dict[str, ArgumentValue]) -> None:
        """Stores the configuration after checking that it builds a solver.

        Args:
            name (str): Unique name of the configuration.
            solver_type (str): One of SOLVER_TYPES.
            arguments (dict[str, ArgumentValue]): Arguments passed to the solver.

        Raises:
            ValueError: If the type is unknown or the solver rejects its arguments.
        """
        if solver_type not in SOLVER_TYPES:
            known = sorted(SOLVER_TYPES)
            raise ValueError(f"Invalid type {solver_type!r} for solver {name!r}. Expected one of {known}.")
        self.name = name
        self.solver_type = solver_type
        self.arguments = dict(arguments)
        self.build()

    def get_name(self) -> str:
        """Returns: str: Unique name of the configuration."""
        return self.name

    def build(self) -> Solver:
        """Returns: Solver: A new solver for this configuration."""
        return SOLVER_TYPES[self.solver_type](self.name, self.arguments)


class ExperimentConfiguration:
    """Holds an experiment: where its instances are, how many seeds each run uses, where results go, and solvers."""

    def __init__(self, instances_dir: Path, seeds: int, out_dir: Path, solvers: list[SolverConfiguration]) -> None:
        """Stores the experiment after checking it is runnable.

        Args:
            instances_dir (Path): Directory of BPPLIB instance files (*.txt).
            seeds (int): Number of seeds per solver configuration and instance (seeds 0 to seeds - 1).
            out_dir (Path): Directory that receives the result files.
            solvers (list[SolverConfiguration]): Solver configurations to compare, with unique names.

        Raises:
            ValueError: If seeds is below 1, there are no solvers, or two solvers share a name.
        """
        if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
            raise ValueError(f"Invalid seeds: {seeds!r}. Expected a positive integer.")
        if not solvers:
            raise ValueError("The experiment lists no solvers. Expected at least one [[solvers]] table.")
        names = [solver.get_name() for solver in solvers]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Solver names {repeated} are repeated. Expected unique names.")
        self.instances_dir = instances_dir
        self.seeds = seeds
        self.out_dir = out_dir
        self.solvers = solvers

    def get_instance_paths(self) -> list[Path]:
        """Returns the instance files of the experiment, sorted by name.

        Returns:
            list[Path]: The *.txt files of the instances directory.

        Raises:
            ValueError: If the directory holds no instance file.
        """
        paths = sorted(self.instances_dir.glob("*.txt"))
        if not paths:
            raise ValueError(f"No instance files (*.txt) in {str(self.instances_dir)!r}.")
        return paths


def read_solver(table: dict[str, object]) -> SolverConfiguration:
    """Builds a solver configuration from one [[solvers]] table: name, type, and the solver's own arguments.

    Args:
        table (dict[str, object]): The parsed table.

    Returns:
        SolverConfiguration: The configuration.

    Raises:
        ValueError: If name or type is missing, or an argument is not a number or a string.
    """
    arguments = dict(table)
    name, solver_type = arguments.pop("name", None), arguments.pop("type", None)
    if not isinstance(name, str) or not isinstance(solver_type, str):
        raise ValueError(f"Invalid solver table {table!r}. Expected string 'name' and 'type' keys.")
    values: dict[str, ArgumentValue] = {}
    for key, value in arguments.items():
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise ValueError(f"Invalid argument {key} = {value!r} for solver {name!r}. Expected a number or a string.")
        values[key] = value
    return SolverConfiguration(name, solver_type, values)


def load_experiment(path: Path) -> ExperimentConfiguration:
    """Reads an experiment from a TOML file; directories are relative to the current working directory.

    Args:
        path (Path): The TOML file, with an [experiment] table (instances, seeds, out) and [[solvers]] tables.

    Returns:
        ExperimentConfiguration: The validated experiment.

    Raises:
        ValueError: If a table or key is missing or unknown, or a value is invalid.
    """
    document = tomllib.loads(path.read_text())
    unknown_tables = sorted(set(document) - {"experiment", "solvers"})
    if unknown_tables:
        raise ValueError(f"Unknown tables {unknown_tables} in {str(path)!r}. Expected 'experiment' and 'solvers'.")
    experiment = document.get("experiment", {})
    if sorted(experiment) != sorted(EXPERIMENT_KEYS):
        found = sorted(experiment)
        raise ValueError(f"Invalid [experiment] keys {found} in {str(path)!r}. Expected {EXPERIMENT_KEYS}.")
    solvers = [read_solver(table) for table in document.get("solvers", [])]
    return ExperimentConfiguration(Path(experiment["instances"]), experiment["seeds"], Path(experiment["out"]), solvers)

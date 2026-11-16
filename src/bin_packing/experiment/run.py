import csv
import time
from collections.abc import Callable, Iterator
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bin_packing.experiment.config import ExperimentConfiguration, SolverConfiguration
from bin_packing.instance.bpplib import read_instance
from bin_packing.instance.lower_bounds import lower_bound_l2
from bin_packing.solvers.assignment_ga.genetic_algorithm import AssignmentGeneticAlgorithm
from bin_packing.solvers.permutation_ga.genetic_algorithm import PermutationGeneticAlgorithm

RESULTS_FILE = "results.csv"
CONVERGENCE_FILE = "convergence.csv"
RESULT_COLUMNS: tuple[str, ...] = (
    "instance",
    "num_items",
    "capacity",
    "solver",
    "seed",
    "bins",
    "feasible",
    "lower_bound",
    "gap",
    "seconds",
)
CONVERGENCE_COLUMNS: tuple[str, ...] = ("instance", "solver", "seed", "generation", "best", "median", "worst")

type Job = tuple[Path, SolverConfiguration, int]
type Row = dict[str, str | int | float | bool]


class RunResult:
    """Holds the result row of one run and, for genetic algorithms, its convergence rows."""

    def __init__(self, row: Row, convergence: list[Row]) -> None:
        """Stores the rows of one run.

        Args:
            row (Row): Values for RESULT_COLUMNS.
            convergence (list[Row]): Values for CONVERGENCE_COLUMNS, one per generation; empty for other solvers.
        """
        self.row = row
        self.convergence = convergence


def run_one(job: Job) -> RunResult:
    """Solves one instance with one solver configuration and one seed, timing the solve.

    Args:
        job (Job): Instance file, solver configuration and seed.

    Returns:
        RunResult: The result row and, for genetic algorithms, the convergence rows.
    """
    path, configuration, seed = job
    instance = read_instance(path)
    solver = configuration.build()
    started = time.perf_counter()
    solution = solver.solve(instance, np.random.default_rng(seed))
    seconds = time.perf_counter() - started
    bound = lower_bound_l2(instance)
    feasible = solution.is_feasible()
    row: Row = {
        "instance": instance.get_name(),
        "num_items": instance.get_num_items(),
        "capacity": instance.get_capacity(),
        "solver": configuration.get_name(),
        "seed": seed,
        "bins": solution.get_num_bins(),
        "feasible": feasible,
        "lower_bound": bound,
        "gap": (solution.get_num_bins() - bound) / bound if feasible else "",
        "seconds": round(seconds, 4),
    }
    convergence: list[Row] = []
    if isinstance(solver, AssignmentGeneticAlgorithm | PermutationGeneticAlgorithm):
        for generation, (best, median, worst) in enumerate(solver.get_history(), start=1):
            convergence.append(
                {
                    "instance": instance.get_name(),
                    "solver": configuration.get_name(),
                    "seed": seed,
                    "generation": generation,
                    "best": best,
                    "median": median,
                    "worst": worst,
                }
            )
    return RunResult(row, convergence)


def run_experiment(
    experiment: ExperimentConfiguration, processes: int | None = None, report: Callable[[int, int], None] | None = None
) -> list[Path]:
    """Runs every solver configuration on every instance with every seed and writes the CSV result files.

    Args:
        experiment (ExperimentConfiguration): The experiment to run.
        processes (int | None): Worker processes; None uses every CPU, 1 runs in this process.
        report (Callable[[int, int], None] | None): Called with (finished runs, total runs) after each run.

    Returns:
        list[Path]: The results file and the convergence file.
    """
    jobs: list[Job] = [
        (path, solver, seed)
        for path in experiment.get_instance_paths()
        for solver in experiment.solvers
        for seed in range(experiment.seeds)
    ]
    experiment.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = experiment.out_dir / RESULTS_FILE
    convergence_path = experiment.out_dir / CONVERGENCE_FILE
    with results_path.open("w", newline="") as results_file, convergence_path.open("w", newline="") as history_file:
        results = csv.DictWriter(results_file, fieldnames=RESULT_COLUMNS)
        history = csv.DictWriter(history_file, fieldnames=CONVERGENCE_COLUMNS)
        results.writeheader()
        history.writeheader()
        for finished, result in enumerate(run_all(jobs, processes), start=1):
            results.writerow(result.row)
            history.writerows(result.convergence)
            if report is not None:
                report(finished, len(jobs))
    return [results_path, convergence_path]


def run_all(jobs: list[Job], processes: int | None) -> Iterator[RunResult]:
    """Yields the result of each job in job order, in this process or in a pool of worker processes.

    Args:
        jobs (list[Job]): Runs to perform.
        processes (int | None): Worker processes; None uses every CPU, 1 runs in this process.

    Yields:
        RunResult: The result of each job, in the order of jobs.
    """
    if processes == 1:
        yield from map(run_one, jobs)
        return
    with Pool(processes) as pool:
        yield from pool.imap(run_one, jobs)

import argparse
import sys
from pathlib import Path

from bin_packing.experiment.config import load_experiment
from bin_packing.experiment.plot import plot_convergence, plot_gaps
from bin_packing.experiment.run import run_experiment
from bin_packing.experiment.summarize import summarize_results
from bin_packing.instance.bpplib import write_instance
from bin_packing.instance.generator import SIZE_SCALINGS, generate_weibull_instance


def generate(arguments: argparse.Namespace) -> None:
    """Writes count Weibull instances with consecutive seeds starting at the given one."""
    for seed in range(arguments.seed, arguments.seed + arguments.count):
        instance = generate_weibull_instance(arguments.num_items, arguments.capacity, arguments.size_scaling, seed)
        print(write_instance(instance, arguments.out))


def run(arguments: argparse.Namespace) -> None:
    """Runs the experiment of a TOML file, reporting progress on one line."""
    experiment = load_experiment(arguments.config)

    def report(finished: int, total: int) -> None:
        print(f"\rRuns finished: {finished}/{total}", end="" if finished < total else "\n", flush=True)

    for path in run_experiment(experiment, arguments.processes, report):
        print(path)


def summarize(arguments: argparse.Namespace) -> None:
    """Prints, and writes next to the results, the Markdown summary of an experiment."""
    print(summarize_results(arguments.results, arguments.reference))


def plot(arguments: argparse.Namespace) -> None:
    """Draws the gap and convergence figures of an experiment next to its results."""
    print(plot_gaps(arguments.results))
    convergence = plot_convergence(arguments.results)
    if convergence is not None:
        print(convergence)


def build_parser() -> argparse.ArgumentParser:
    """Returns: argparse.ArgumentParser: The parser of the four subcommands."""
    parser = argparse.ArgumentParser(prog="bin_packing", description="Bin packing instances, solvers and experiments.")
    commands = parser.add_subparsers(dest="command", required=True)

    generating = commands.add_parser("generate", help="write Weibull instances in BPPLIB format")
    generating.add_argument("--num-items", type=int, required=True)
    generating.add_argument("--capacity", type=int, required=True)
    generating.add_argument("--count", type=int, default=1, help="instances to write, with consecutive seeds")
    generating.add_argument("--size-scaling", choices=SIZE_SCALINGS, default="scaled")
    generating.add_argument("--seed", type=int, default=0, help="seed of the first instance")
    generating.add_argument("--out", type=Path, required=True, help="directory that receives the files")
    generating.set_defaults(handler=generate)

    running = commands.add_parser("run", help="run the experiment described by a TOML file")
    running.add_argument("--config", type=Path, required=True)
    running.add_argument("--processes", type=int, default=None, help="worker processes (default: every CPU)")
    running.set_defaults(handler=run)

    summarizing = commands.add_parser("summarize", help="tabulate an experiment's results")
    summarizing.add_argument("results", type=Path, help="directory holding results.csv")
    summarizing.add_argument("--reference", default="FFD", help="solver the others are compared with")
    summarizing.set_defaults(handler=summarize)

    plotting = commands.add_parser("plot", help="draw an experiment's figures")
    plotting.add_argument("results", type=Path, help="directory holding results.csv")
    plotting.set_defaults(handler=plot)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs one subcommand, reporting invalid input as an error message instead of a traceback.

    Args:
        argv (list[str] | None): Command-line arguments; None reads them from sys.argv.

    Returns:
        int: The exit status, 0 on success and 1 on invalid input.
    """
    arguments = build_parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

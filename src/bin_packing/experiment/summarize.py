import csv
from pathlib import Path

from bin_packing.experiment.run import RESULTS_FILE

SUMMARY_FILE = "summary.md"


def read_results(results_dir: Path) -> list[dict[str, str]]:
    """Reads the rows of the results file of an experiment.

    Args:
        results_dir (Path): Directory holding results.csv.

    Returns:
        list[dict[str, str]]: One row per run, as written by run_experiment.

    Raises:
        ValueError: If the results file is missing or empty.
    """
    path = results_dir / RESULTS_FILE
    if not path.is_file():
        raise ValueError(f"Missing results file {str(path)!r}. Run the experiment first.")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"The results file {str(path)!r} has no runs.")
    return rows


def compare_with_reference(rows: list[dict[str, str]], reference_bins: dict[tuple[str, str], int]) -> str:
    """Returns: str: Counts "better/equal/worse" of feasible runs against the reference run with the same instance
    and seed."""
    outcomes = [
        int(row["bins"]) - reference_bins[(row["instance"], row["seed"])]
        for row in rows
        if row["feasible"] == "True" and (row["instance"], row["seed"]) in reference_bins
    ]
    better = sum(1 for difference in outcomes if difference < 0)
    worse = sum(1 for difference in outcomes if difference > 0)
    return f"{better}/{len(outcomes) - better - worse}/{worse}"


def summarize_group(rows: list[dict[str, str]], reference_bins: dict[tuple[str, str], int]) -> list[str]:
    """Returns: list[str]: The table cells for the runs of one solver at one number of items."""
    feasible = [row for row in rows if row["feasible"] == "True"]
    at_bound = sum(1 for row in feasible if row["bins"] == row["lower_bound"])
    # Showing a dash when no run was feasible: the gap is only defined for feasible packings.
    mean_gap = f"{100 * sum(float(row['gap']) for row in feasible) / len(feasible):.2f}%" if feasible else "—"
    mean_seconds = sum(float(row["seconds"]) for row in rows) / len(rows)
    return [
        str(len(rows)),
        str(len(rows) - len(feasible)),
        f"{at_bound}/{len(rows)}",
        mean_gap,
        compare_with_reference(rows, reference_bins),
        f"{mean_seconds:.3f}",
    ]


def summarize_results(results_dir: Path, reference: str = "FFD") -> str:
    """Builds a Markdown table per number of items comparing every solver configuration.

    Args:
        results_dir (Path): Directory holding results.csv; summary.md is written there too.
        reference (str): Solver configuration the others are compared with, run by run.

    Returns:
        str: The Markdown summary that was written.
    """
    rows = read_results(results_dir)
    reference_bins = {(row["instance"], row["seed"]): int(row["bins"]) for row in rows if row["solver"] == reference}
    solvers = list(dict.fromkeys(row["solver"] for row in rows))
    lines = []
    for num_items in sorted({int(row["num_items"]) for row in rows}):
        lines += [
            f"### {num_items} items",
            "",
            f"| Solver | Runs | Infeasible | At L2 | Mean gap | vs {reference} (better/equal/worse) | Mean seconds |",
            "|---|---|---|---|---|---|---|",
        ]
        for solver in solvers:
            group = [row for row in rows if row["solver"] == solver and int(row["num_items"]) == num_items]
            if group:
                lines.append("| " + " | ".join([solver, *summarize_group(group, reference_bins)]) + " |")
        lines.append("")
    summary = "\n".join(lines)
    (results_dir / SUMMARY_FILE).write_text(summary)
    return summary

import csv
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bin_packing.experiment.run import CONVERGENCE_FILE
from bin_packing.experiment.summarize import read_results

GAP_FIGURE = "gap_by_solver.png"
CONVERGENCE_FIGURE = "convergence.png"
# Using the light chart surface, text inks and categorical order of the dataviz reference palette; the categorical
# hues were validated for colour-vision deficiency, and the two below 3:1 contrast are relieved by direct labels.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def style_axes(axes: Axes) -> None:
    """Applies the recessive grid, surface and text colours shared by every chart."""
    axes.set_facecolor(SURFACE)
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(colors=TEXT_SECONDARY, length=0)


def describe_gap(rows: list[dict[str, str]]) -> tuple[float, str]:
    """Returns the mean gap (%) of a solver's feasible runs and the label printed next to its bar.

    Args:
        rows (list[dict[str, str]]): Runs of one solver at one number of items.

    Returns:
        tuple[float, str]: The mean gap, 0 when no run is feasible, and a label such as "0.54%" or
            "8.58% (24/30 infeasible)".
    """
    feasible = [100 * float(row["gap"]) for row in rows if row["feasible"] == "True"]
    if not feasible:
        return 0.0, "no feasible run"
    mean_gap = sum(feasible) / len(feasible)
    infeasible = len(rows) - len(feasible)
    return mean_gap, f"{mean_gap:.2f}%" + (f"  ({infeasible}/{len(rows)} infeasible)" if infeasible else "")


def plot_gaps(results_dir: Path) -> Path:
    """Draws one panel per number of items with each solver's mean gap to the L2 bound as a labelled bar.

    Args:
        results_dir (Path): Directory holding results.csv; the figure is written there too.

    Returns:
        Path: The written figure.
    """
    rows = read_results(results_dir)
    solvers = list(dict.fromkeys(row["solver"] for row in rows))
    sizes = sorted({int(row["num_items"]) for row in rows})
    # Drawing on a bare Figure, without pyplot, so no display backend is needed.
    figure = Figure(figsize=(6 * len(sizes), 0.42 * len(solvers) + 1.4))
    panels = figure.subplots(1, len(sizes), squeeze=False, sharey=True)
    figure.patch.set_facecolor(SURFACE)
    for axes, num_items in zip(panels[0], sizes, strict=True):
        described = [
            describe_gap([row for row in rows if row["solver"] == solver and int(row["num_items"]) == num_items])
            for solver in solvers
        ]
        gaps = [gap for gap, _ in described]
        axes.barh(range(len(solvers)), gaps, height=0.6, color=SERIES[0])
        for position, (gap, label) in enumerate(described):
            axes.annotate(
                label,
                (gap, position),
                xytext=(4, 0),
                textcoords="offset points",
                color=TEXT_PRIMARY,
                fontsize=8,
                va="center",
            )
        axes.set_yticks(range(len(solvers)), solvers)
        axes.set_xlim(right=max(gaps) * 1.6)
        style_axes(axes)
        axes.set_title(f"{num_items} items", color=TEXT_PRIMARY, loc="left")
        axes.set_xlabel("Mean gap to the L2 lower bound (%)", color=TEXT_SECONDARY)
    # Inverting the shared y axis once puts the first solver on top; inverting it per panel would undo it.
    panels[0][0].invert_yaxis()
    figure.tight_layout()
    path = results_dir / GAP_FIGURE
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def plot_convergence(results_dir: Path) -> Path | None:
    """Draws the best fitness per generation of each genetic algorithm on the first instance, seed 0.

    Args:
        results_dir (Path): Directory holding convergence.csv; the figure is written there too.

    Returns:
        Path | None: The written figure, or None when the experiment ran no genetic algorithm or its
            convergence file was moved away.

    Raises:
        ValueError: If more genetic algorithm configurations ran than there are categorical colours.
    """
    convergence_path = results_dir / CONVERGENCE_FILE
    if not convergence_path.is_file():
        return None
    with convergence_path.open(newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["seed"] == "0"]
    if not rows:
        return None
    instance = min(row["instance"] for row in rows)
    rows = [row for row in rows if row["instance"] == instance]
    solvers = list(dict.fromkeys(row["solver"] for row in rows))
    if len(solvers) > len(SERIES):
        raise ValueError(f"{len(solvers)} genetic algorithms ran; the chart distinguishes at most {len(SERIES)}.")
    figure = Figure(figsize=(8, 4.5))
    axes = figure.subplots()
    figure.patch.set_facecolor(SURFACE)
    for colour, solver in zip(SERIES, solvers, strict=False):
        history = [row for row in rows if row["solver"] == solver]
        generations = [int(row["generation"]) for row in history]
        best = [float(row["best"]) for row in history]
        axes.plot(generations, best, color=colour, linewidth=2, label=solver)
        axes.annotate(
            solver,
            (generations[-1], best[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=TEXT_PRIMARY,
            fontsize=8,
            va="center",
        )
    style_axes(axes)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_title(f"Best fitness per generation, instance {instance}, seed 0", color=TEXT_PRIMARY, loc="left")
    axes.set_xlabel("Generation", color=TEXT_SECONDARY)
    axes.set_ylabel("Falkenauer fitness", color=TEXT_SECONDARY)
    # Leaving room on the right for the direct labels at the line ends.
    axes.set_xlim(right=max(int(row["generation"]) for row in rows) * 1.25)
    # Naming a single series in the title is enough; two or more also need a legend.
    if len(solvers) > 1:
        axes.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="upper left", bbox_to_anchor=(0, -0.15), ncols=2)
    figure.tight_layout()
    path = results_dir / CONVERGENCE_FIGURE
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    return path

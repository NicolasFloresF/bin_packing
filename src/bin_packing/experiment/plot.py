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


def plot_gaps(results_dir: Path) -> Path:
    """Draws one panel per number of items with the distribution of each solver's gap to the L2 bound.

    Args:
        results_dir (Path): Directory holding results.csv; the figure is written there too.

    Returns:
        Path: The written figure.
    """
    rows = [row for row in read_results(results_dir) if row["feasible"] == "True"]
    solvers = list(dict.fromkeys(row["solver"] for row in rows))
    sizes = sorted({int(row["num_items"]) for row in rows})
    # Drawing on a bare Figure, without pyplot, so no display backend is needed.
    figure = Figure(figsize=(5.5 * len(sizes), 0.5 * len(solvers) + 1.5))
    panels = figure.subplots(1, len(sizes), squeeze=False)
    figure.patch.set_facecolor(SURFACE)
    for axes, num_items in zip(panels[0], sizes, strict=True):
        gaps = [
            [100 * float(row["gap"]) for row in rows if row["solver"] == solver and int(row["num_items"]) == num_items]
            for solver in solvers
        ]
        axes.boxplot(
            gaps,
            orientation="horizontal",
            tick_labels=solvers,
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": SERIES[0], "edgecolor": SERIES[0], "alpha": 0.35},
            medianprops={"color": TEXT_PRIMARY, "linewidth": 2},
            whiskerprops={"color": TEXT_SECONDARY},
            capprops={"color": TEXT_SECONDARY},
            flierprops={"marker": "o", "markersize": 4, "markerfacecolor": TEXT_SECONDARY, "markeredgecolor": SURFACE},
        )
        axes.invert_yaxis()
        style_axes(axes)
        axes.set_title(f"{num_items} items", color=TEXT_PRIMARY, loc="left")
        axes.set_xlabel("Gap to the L2 lower bound (%)", color=TEXT_SECONDARY)
    figure.tight_layout()
    path = results_dir / GAP_FIGURE
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def plot_convergence(results_dir: Path) -> Path | None:
    """Draws the best fitness per generation of each genetic algorithm on the first instance, seed 0.

    Args:
        results_dir (Path): Directory holding convergence.csv; the figure is written there too.

    Returns:
        Path | None: The written figure, or None when the experiment ran no genetic algorithm.

    Raises:
        ValueError: If more genetic algorithm configurations ran than there are categorical colours.
    """
    with (results_dir / CONVERGENCE_FILE).open(newline="") as file:
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
    axes.set_title(f"Best fitness per generation — {instance}, seed 0", color=TEXT_PRIMARY, loc="left")
    axes.set_xlabel("Generation", color=TEXT_SECONDARY)
    axes.set_ylabel("Falkenauer fitness", color=TEXT_SECONDARY)
    # Leaving room on the right for the direct labels at the line ends.
    axes.set_xlim(right=max(int(row["generation"]) for row in rows) * 1.25)
    # Naming a single series in the title is enough; two or more also need a legend.
    if len(solvers) > 1:
        axes.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="lower right")
    figure.tight_layout()
    path = results_dir / CONVERGENCE_FIGURE
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    return path

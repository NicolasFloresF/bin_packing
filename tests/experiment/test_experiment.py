import csv
from pathlib import Path

import pytest

from bin_packing.__main__ import main
from bin_packing.experiment.config import load_experiment
from bin_packing.experiment.plot import plot_convergence, plot_gaps
from bin_packing.experiment.run import run_experiment
from bin_packing.experiment.summarize import summarize_results

SOLVERS = """
[[solvers]]
name = "FFD"
type = "heuristic"

[[solvers]]
name = "WFD"
type = "heuristic"
bin_selection_rule = "worst"

[[solvers]]
name = "AGA"
type = "assignment_ga"
generations = 5

[[solvers]]
name = "PGA"
type = "permutation_ga"
generations = 5

[[solvers]]
name = "SA"
type = "simulated_annealing"
iterations_per_temperature = 20
"""


def write_config(tmp_path: Path, solvers: str = SOLVERS, seeds: str = "2") -> Path:
    """Returns: Path: A TOML experiment over tmp_path/instances, writing to tmp_path/results."""
    path = tmp_path / "experiment.toml"
    experiment = (
        f'[experiment]\ninstances = "{tmp_path / "instances"}"\nseeds = {seeds}\nout = "{tmp_path / "results"}"\n'
    )
    path.write_text(experiment + solvers)
    return path


def test_cli_generates_runs_summarizes_and_plots(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    instances = str(tmp_path / "instances")
    assert main(["generate", "--num-items", "30", "--capacity", "150", "--count", "2", "--out", instances]) == 0
    assert sorted(path.name for path in (tmp_path / "instances").iterdir()) == [
        "weibull_scaled_n30_c150_s0.txt",
        "weibull_scaled_n30_c150_s1.txt",
    ]
    assert main(["run", "--config", str(write_config(tmp_path)), "--processes", "1"]) == 0
    with (tmp_path / "results" / "results.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2 * 5 * 2
    assert {row["solver"] for row in rows} == {"FFD", "WFD", "AGA", "PGA", "SA"}
    assert main(["summarize", str(tmp_path / "results")]) == 0
    assert "| FFD | 4 | 0 |" in capsys.readouterr().out
    assert main(["plot", str(tmp_path / "results")]) == 0
    assert (tmp_path / "results" / "gap_by_solver.png").is_file()


def test_convergence_rows_come_only_from_genetic_algorithms(tmp_path: Path) -> None:
    main(["generate", "--num-items", "30", "--capacity", "150", "--seed", "3", "--out", str(tmp_path / "instances")])
    experiment = load_experiment(write_config(tmp_path, seeds="1"))
    results_path, convergence_path = run_experiment(experiment, processes=1)
    with convergence_path.open(newline="") as file:
        solvers = {row["solver"] for row in csv.DictReader(file)}
    assert solvers <= {"AGA", "PGA"}
    assert "AGA" in solvers
    assert results_path.is_file()
    assert plot_convergence(tmp_path / "results") == tmp_path / "results" / "convergence.png"


def test_experiment_without_genetic_algorithms_draws_no_convergence(tmp_path: Path) -> None:
    main(["generate", "--num-items", "20", "--capacity", "150", "--out", str(tmp_path / "instances")])
    experiment = load_experiment(write_config(tmp_path, solvers='[[solvers]]\nname = "FFD"\ntype = "heuristic"\n'))
    run_experiment(experiment, processes=2)
    assert plot_convergence(tmp_path / "results") is None
    assert plot_gaps(tmp_path / "results").is_file()
    summary = summarize_results(tmp_path / "results")
    assert "| FFD | 2 | 0 |" in summary
    assert (tmp_path / "results" / "summary.md").read_text() == summary


@pytest.mark.parametrize(
    ("solvers", "message"),
    [
        ('[[solvers]]\nname = "X"\ntype = "tabu"\n', "Invalid type 'tabu'"),
        ('[[solvers]]\nname = "X"\ntype = "heuristic"\nrule = "first"\n', r"Unknown arguments \['rule'\]"),
        ('[[solvers]]\nname = "X"\ntype = "heuristic"\n[[solvers]]\nname = "X"\ntype = "heuristic"\n', "repeated"),
        ('[[solvers]]\ntype = "heuristic"\n', "'name' and 'type'"),
        ('[[solvers]]\nname = "X"\ntype = "permutation_ga"\ngenerations = true\n', "number or a string"),
        ("", "no solvers"),
        ('[extra]\n[[solvers]]\nname = "X"\ntype = "heuristic"\n', "Unknown tables"),
    ],
)
def test_invalid_experiments_are_rejected(tmp_path: Path, solvers: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_experiment(write_config(tmp_path, solvers=solvers))


def test_invalid_input_is_reported_without_a_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "instances").mkdir()
    assert main(["run", "--config", str(write_config(tmp_path, seeds="0"))]) == 1
    assert "error: Invalid seeds: 0" in capsys.readouterr().err
    assert main(["summarize", str(tmp_path / "missing")]) == 1
    config = write_config(tmp_path)
    experiment_text = config.read_text().replace("seeds = 2\n", "")
    config.write_text(experiment_text)
    assert main(["run", "--config", str(config)]) == 1
    assert main(["run", "--config", str(write_config(tmp_path))]) == 1
    assert "No instance files" in capsys.readouterr().err

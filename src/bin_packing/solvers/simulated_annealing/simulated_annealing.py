import math

import numpy as np
from numpy.random import Generator

from bin_packing.instance.instance import Instance
from bin_packing.instance.lower_bounds import lower_bound_l2
from bin_packing.solvers.arguments import (
    Arguments,
    merge_arguments,
    require_choice,
    require_number,
    require_positive_int,
)
from bin_packing.solvers.heuristics.bin_selection import pack
from bin_packing.solvers.simulated_annealing.packing import Packing
from bin_packing.solvers.solution import Solution
from bin_packing.solvers.solver import Solver

INITIALIZATIONS = ("random", "ffd_seeded")
# Choosing a swap of two items or a move of one item with equal probability at each iteration.
SWAP_PROBABILITY = 0.5

# Cooling from 1e-5 to 1e-8 in 135 levels of 7 500 neighbours (about 1M iterations, ~1 s at 500 items). Measured on
# 120- and 500-item Weibull instances, 1e-5 balances both sizes: hotter starts helped at 120 items and hurt at 500.
DEFAULT_ARGUMENTS: Arguments = {
    "initial_temperature": 1e-5,
    "final_temperature": 1e-8,
    "cooling_rate": 0.95,
    "iterations_per_temperature": 7_500,
    "initialization": "random",
}


class BestPacking:
    """Keeps a copy of the fittest packing seen during a run."""

    def __init__(self, packing: Packing) -> None:
        """Starts from the given packing.

        Args:
            packing (Packing): The initial packing.
        """
        self.fitness = packing.get_fitness()
        self.bins = packing.get_used_bins()

    def offer(self, packing: Packing) -> None:
        """Copies the packing if it is fitter than the best one kept."""
        if packing.get_fitness() > self.fitness:
            self.fitness = packing.get_fitness()
            self.bins = packing.get_used_bins()

    def get_bins(self) -> list[list[int]]:
        """Returns: list[list[int]]: Item indices of each bin of the best packing."""
        return self.bins


class SimulatedAnnealing(Solver):
    """Improves one feasible packing by random moves and swaps of items, accepting worse ones less often as it cools.

    The temperature starts at initial_temperature, stays fixed for iterations_per_temperature neighbours, then is
    multiplied by cooling_rate, until it falls below final_temperature.
    """

    def __init__(self, name: str, arguments: Arguments) -> None:
        """Builds the simulated annealing from named arguments; omitted ones take DEFAULT_ARGUMENTS.

        Args:
            name (str): Unique name of the configuration.
            arguments (Arguments): Any of initial_temperature, final_temperature, cooling_rate,
                iterations_per_temperature and initialization ("random" or "ffd_seeded").

        Raises:
            ValueError: If an argument is unknown or has an invalid value, the final temperature is not below the
                initial one, or the cooling rate is not strictly between 0 and 1.
        """
        super().__init__(name)
        merged = merge_arguments(name, arguments, DEFAULT_ARGUMENTS)
        self.initial_temperature = require_number(merged, "initial_temperature", 0.0, math.inf)
        self.final_temperature = require_number(merged, "final_temperature", 0.0, math.inf)
        self.cooling_rate = require_number(merged, "cooling_rate", 0.0, 1.0)
        self.iterations_per_temperature = require_positive_int(merged, "iterations_per_temperature")
        self.initialization = require_choice(merged, "initialization", INITIALIZATIONS)
        if not 0.0 < self.final_temperature < self.initial_temperature:
            raise ValueError(
                f"Invalid final_temperature: {self.final_temperature!r}. "
                f"Expected above 0 and below initial_temperature ({self.initial_temperature!r})."
            )
        if not 0.0 < self.cooling_rate < 1.0:
            raise ValueError(f"Invalid cooling_rate: {self.cooling_rate!r}. Expected strictly between 0 and 1.")

    def get_num_levels(self) -> int:
        """Returns: int: Number of temperatures visited, from the initial one down to the last not below the final."""
        return math.floor(math.log(self.final_temperature / self.initial_temperature) / math.log(self.cooling_rate)) + 1

    def solve(self, instance: Instance, rng: Generator) -> Solution:
        """Anneals from a First Fit packing until the temperature falls below the final one or the packing reaches L2.

        Args:
            instance (Instance): Instance to solve.
            rng (Generator): The only source of randomness of the run.

        Returns:
            Solution: The fittest packing visited; always feasible.
        """
        bound = lower_bound_l2(instance)
        packing = Packing(instance, self.initial_bins(instance, rng))
        best = BestPacking(packing)
        temperature = self.initial_temperature
        for _ in range(self.get_num_levels()):
            if packing.get_num_used_bins() == bound:
                break
            self.anneal_at(packing, temperature, best, bound, rng)
            temperature *= self.cooling_rate
        return Solution(instance, best.get_bins())

    def anneal_at(self, packing: Packing, temperature: float, best: BestPacking, bound: int, rng: Generator) -> None:
        """Tries iterations_per_temperature neighbours at one temperature, stopping early if the packing reaches L2.

        Args:
            packing (Packing): Current packing, changed in place by accepted steps.
            temperature (float): Temperature of this level.
            best (BestPacking): Fittest packing so far, updated after accepted steps.
            bound (int): L2 lower bound of the instance; reaching it proves the packing optimal.
            rng (Generator): Source of randomness.
        """
        count = self.iterations_per_temperature
        swaps = rng.random(count) < SWAP_PROBABILITY
        items = rng.integers(packing.get_num_items(), size=count)
        other_items = rng.integers(packing.get_num_items(), size=count)
        targets = rng.integers(packing.get_num_bins(), size=count)
        # Accepting a change of fitness c with probability exp(c / T) is accepting it when c >= T * ln(u).
        tolerances = temperature * np.log(1.0 - rng.random(count))
        for iteration in range(count):
            swap = bool(swaps[iteration])
            # Reading the second number as the other item of a swap, or as the target bin of a move.
            partner = int(other_items[iteration]) if swap else int(targets[iteration])
            if self.step(packing, int(items[iteration]), partner, swap, float(tolerances[iteration])):
                best.offer(packing)
                if packing.get_num_used_bins() == bound:
                    return

    def step(self, packing: Packing, item: int, partner: int, swap: bool, tolerance: float) -> bool:
        """Tries one swap of item with partner, or one move of item to bin partner, and applies it if accepted.

        Args:
            packing (Packing): Current packing, changed in place when the step is accepted.
            item (int): Item to swap or move.
            partner (int): Other item of the swap, or target bin of the move.
            swap (bool): True for a swap, False for a move.
            tolerance (float): Largest fitness loss (as a non-positive change) accepted at this iteration.

        Returns:
            bool: True if the step was feasible and accepted.
        """
        candidate = packing.fitness_after_swap(item, partner) if swap else packing.fitness_after_move(item, partner)
        if candidate is None or candidate - packing.get_fitness() < tolerance:
            return False
        if swap:
            packing.apply_swap(item, partner)
        else:
            packing.apply_move(item, partner)
        return True

    def initial_bins(self, instance: Instance, rng: Generator) -> list[list[int]]:
        """Returns: list[list[int]]: First Fit over a random order, or over decreasing sizes for "ffd_seeded"."""
        if self.initialization == "random":
            order = [int(item) for item in rng.permutation(instance.get_num_items())]
        else:
            sizes = instance.get_item_sizes()
            order = sorted(range(instance.get_num_items()), key=lambda item: -sizes[item])
        return pack(order, instance, "first", rng)

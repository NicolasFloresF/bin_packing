from numpy.random import Generator

from bin_packing.instance.instance import Instance
from bin_packing.solvers.heuristics.bin_selection import BIN_SELECTION_RULES, pack
from bin_packing.solvers.solution import Solution
from bin_packing.solvers.solver import Solver


class DecreasingFitHeuristic(Solver):
    """Packs items from largest to smallest, placing each in a bin chosen by a bin selection rule."""

    def __init__(self, name: str, bin_selection_rule: str) -> None:
        """Builds the heuristic for one bin selection rule.

        Args:
            name (str): Unique name of the configuration, such as "FFD".
            bin_selection_rule (str): One of "first", "best", "worst" or "random".

        Raises:
            ValueError: If the rule is unknown.
        """
        super().__init__(name)
        if bin_selection_rule not in BIN_SELECTION_RULES:
            raise ValueError(
                f"Invalid bin selection rule: {bin_selection_rule!r}. Expected one of {tuple(BIN_SELECTION_RULES)}."
            )
        self.bin_selection_rule = bin_selection_rule

    def solve(self, instance: Instance, rng: Generator) -> Solution:
        """Packs the items in decreasing size order; equal sizes keep their index order.

        Args:
            instance (Instance): Instance to solve.
            rng (Generator): Source of randomness, used only by the "random" rule.

        Returns:
            Solution: A feasible packing.
        """
        sizes = instance.get_item_sizes()
        order = sorted(range(instance.get_num_items()), key=lambda item: -sizes[item])
        return Solution(instance, pack(order, instance, self.bin_selection_rule, rng))

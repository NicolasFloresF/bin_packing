from abc import ABC, abstractmethod

from numpy.random import Generator

from bin_packing.instance.instance import Instance
from bin_packing.solvers.solution import Solution


class Solver(ABC):
    """Defines a solver configuration: a method with fixed parameters that packs an instance."""

    def __init__(self, name: str) -> None:
        """Stores the name that identifies this solver configuration in results.

        Args:
            name (str): Unique name of the configuration, such as "FFD".

        Raises:
            ValueError: If the name is empty.
        """
        if not name:
            raise ValueError(f"Invalid solver name: {name!r}. Expected a non-empty string.")
        self.name = name

    def get_name(self) -> str:
        """Returns: str: Name that identifies this solver configuration."""
        return self.name

    @abstractmethod
    def solve(self, instance: Instance, rng: Generator) -> Solution:
        """Packs every item of an instance into bins.

        Args:
            instance (Instance): Instance to solve.
            rng (Generator): The only source of randomness of the run.

        Returns:
            Solution: The packing found.
        """

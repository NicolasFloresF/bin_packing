from numpy.random import Generator

from bin_packing.instance.instance import Instance
from bin_packing.instance.lower_bounds import lower_bound_l2
from bin_packing.solvers.arguments import (
    Arguments,
    merge_arguments,
    require_choice,
    require_optional_number,
    require_positive_int,
)
from bin_packing.solvers.fitness.falkenauer import falkenauer_fitness
from bin_packing.solvers.heuristics.bin_selection import pack
from bin_packing.solvers.permutation_ga.operators import CROSSOVERS, INITIALIZATIONS, MUTATIONS, SELECTIONS
from bin_packing.solvers.solution import Solution
from bin_packing.solvers.solver import Solver

DEFAULT_ARGUMENTS: Arguments = {
    "population_size": 50,
    "generations": 1000,
    "offspring_per_generation": 50,
    "initialization": "random",
    "selection": "tournament",
    "crossover": "order",
    "mutation": "swap",
    "mutation_rate": None,
}


class Individual:
    """Holds an item order with the First Fit packing it decodes to and that packing's fitness."""

    def __init__(self, order: list[int], bins: list[list[int]], fitness: float) -> None:
        """Stores an evaluated item order.

        Args:
            order (list[int]): Item indices in packing order.
            bins (list[list[int]]): Item indices per bin after First Fit.
            fitness (float): Falkenauer fitness of the packing.
        """
        self.order = order
        self.bins = bins
        self.fitness = fitness


def rank(population: list[Individual]) -> list[Individual]:
    """Returns: list[Individual]: The individuals from fittest to least fit; ties keep their order."""
    return sorted(population, key=lambda individual: -individual.fitness)


class PermutationGeneticAlgorithm(Solver):
    """Evolves item orders, each turned into a packing by First Fit, so every candidate is feasible."""

    def __init__(self, name: str, arguments: Arguments) -> None:
        """Builds the genetic algorithm from named arguments; omitted ones take DEFAULT_ARGUMENTS.

        Args:
            name (str): Unique name of the configuration.
            arguments (Arguments): Any of population_size, generations, offspring_per_generation, initialization
                ("random" or "ffd_seeded"), selection, crossover, mutation and mutation_rate (None for 1/n).

        Raises:
            ValueError: If an argument is unknown or has an invalid value.
        """
        super().__init__(name)
        merged = merge_arguments(name, arguments, DEFAULT_ARGUMENTS)
        self.population_size = require_positive_int(merged, "population_size")
        self.generations = require_positive_int(merged, "generations")
        self.offspring_per_generation = require_positive_int(merged, "offspring_per_generation")
        self.initialize = INITIALIZATIONS[require_choice(merged, "initialization", INITIALIZATIONS)]
        self.select = SELECTIONS[require_choice(merged, "selection", SELECTIONS)]
        self.cross = CROSSOVERS[require_choice(merged, "crossover", CROSSOVERS)]
        self.mutate = MUTATIONS[require_choice(merged, "mutation", MUTATIONS)]
        self.mutation_rate = require_optional_number(merged, "mutation_rate", 0.0, 1.0)
        self.history: list[tuple[float, float, float]] = []

    def get_history(self) -> list[tuple[float, float, float]]:
        """Returns: list[tuple[float, float, float]]: Best, median and worst fitness per generation of the last run."""
        return list(self.history)

    def solve(self, instance: Instance, rng: Generator) -> Solution:
        """Evolves the population until the generations run out or an individual reaches the L2 bound.

        Args:
            instance (Instance): Instance to solve.
            rng (Generator): The only source of randomness of the run.

        Returns:
            Solution: The packing of the fittest individual of the final population.
        """
        bound = lower_bound_l2(instance)
        rate = self.mutation_rate if self.mutation_rate is not None else 1 / instance.get_num_items()
        initial = self.initialize(instance, self.population_size, rng)
        population = rank([self.evaluate(order, instance, rng) for order in initial])
        self.history = []
        for _ in range(self.generations):
            if any(len(individual.bins) == bound for individual in population):
                break
            children = self.breed(population, rate, rng)
            evaluated = [self.evaluate(child, instance, rng) for child in children]
            population = rank(population + evaluated)[: self.population_size]
            self.history.append(
                (population[0].fitness, population[len(population) // 2].fitness, population[-1].fitness)
            )
        return Solution(instance, population[0].bins)

    def breed(self, population: list[Individual], rate: float, rng: Generator) -> list[list[int]]:
        """Returns: list[list[int]]: offspring_per_generation mutated children, each from two selected parents."""
        fitnesses = [individual.fitness for individual in population]
        children = []
        for _ in range(self.offspring_per_generation):
            first = population[self.select(fitnesses, rng)].order
            second = population[self.select(fitnesses, rng)].order
            children.append(self.mutate(self.cross(first, second, rng), rate, rng))
        return children

    def evaluate(self, order: list[int], instance: Instance, rng: Generator) -> Individual:
        """Returns: Individual: The order with its First Fit packing and that packing's Falkenauer fitness."""
        bins = pack(order, instance, "first", rng)
        sizes = instance.get_item_sizes()
        loads = [sum(sizes[item] for item in items) for items in bins]
        return Individual(order, bins, falkenauer_fitness(loads, instance.get_capacity()))

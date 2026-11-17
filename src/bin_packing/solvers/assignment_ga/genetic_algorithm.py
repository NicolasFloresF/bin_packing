import math

from numpy.random import Generator

from bin_packing.instance.instance import Instance
from bin_packing.instance.lower_bounds import lower_bound_l2
from bin_packing.solvers.arguments import (
    Arguments,
    merge_arguments,
    require_choice,
    require_number,
    require_optional_number,
    require_positive_int,
)
from bin_packing.solvers.assignment_ga.operators import (
    CROSSOVERS,
    MUTATIONS,
    SELECTIONS,
    compute_loads,
    compute_penalized_fitness,
    create_random_individual,
)
from bin_packing.solvers.solution import Solution
from bin_packing.solvers.solver import Solver

# Giving the chromosome 10% more bins than the L2 bound: measured on 120-item Weibull instances, tighter limits
# (such as the FFD bin count) left even the initial population infeasible, and looser ones raised the gap.
BIN_LIMIT_SLACK = 1.1

DEFAULT_ARGUMENTS: Arguments = {
    "population_size": 50,
    "generations": 1000,
    "offspring_per_generation": 50,
    "selection": "roulette_wheel",
    "crossover": "one_point",
    "mutation": "reassign",
    "mutation_rate": None,
    "penalty_severity": 5.0,
}


class Individual:
    """Holds a chromosome with the bin loads and fitness it was evaluated to."""

    def __init__(self, chromosome: list[int], loads: list[int], fitness: float) -> None:
        """Stores an evaluated chromosome.

        Args:
            chromosome (list[int]): Bin index of each item.
            loads (list[int]): Total item size of each bin.
            fitness (float): Penalized fitness of the chromosome.
        """
        self.chromosome = chromosome
        self.loads = loads
        self.fitness = fitness

    def is_feasible(self, capacity: int) -> bool:
        """Returns: bool: True if no bin holds more than the capacity."""
        return all(load <= capacity for load in self.loads)

    def get_num_used_bins(self) -> int:
        """Returns: int: Number of bins holding at least one item."""
        return sum(1 for load in self.loads if load > 0)


def rank(population: list[Individual]) -> list[Individual]:
    """Returns: list[Individual]: The individuals from fittest to least fit; ties keep their order."""
    return sorted(population, key=lambda individual: -individual.fitness)


class AssignmentGeneticAlgorithm(Solver):
    """Evolves chromosomes that assign each item to one of a fixed number of bins, penalizing overloaded bins."""

    def __init__(self, name: str, arguments: Arguments) -> None:
        """Builds the genetic algorithm from named arguments; omitted ones take DEFAULT_ARGUMENTS.

        Args:
            name (str): Unique name of the configuration.
            arguments (Arguments): Any of population_size, generations, offspring_per_generation, selection,
                crossover, mutation, mutation_rate (None for 1/n) and penalty_severity (1 = graduation GA).

        Raises:
            ValueError: If an argument is unknown or has an invalid value.
        """
        super().__init__(name)
        merged = merge_arguments(name, arguments, DEFAULT_ARGUMENTS)
        self.population_size = require_positive_int(merged, "population_size")
        self.generations = require_positive_int(merged, "generations")
        self.offspring_per_generation = require_positive_int(merged, "offspring_per_generation")
        self.select = SELECTIONS[require_choice(merged, "selection", SELECTIONS)]
        self.cross = CROSSOVERS[require_choice(merged, "crossover", CROSSOVERS)]
        self.mutate = MUTATIONS[require_choice(merged, "mutation", MUTATIONS)]
        self.mutation_rate = require_optional_number(merged, "mutation_rate", 0.0, 1.0)
        self.penalty_severity = require_number(merged, "penalty_severity", 0.0, math.inf)
        self.history: list[tuple[float, float, float]] = []

    def get_history(self) -> list[tuple[float, float, float]]:
        """Returns: list[tuple[float, float, float]]: Best, median and worst fitness per generation of the last run."""
        return list(self.history)

    def solve(self, instance: Instance, rng: Generator) -> Solution:
        """Evolves the population until the generations run out or a feasible individual reaches the L2 bound.

        Args:
            instance (Instance): Instance to solve.
            rng (Generator): The only source of randomness of the run.

        Returns:
            Solution: The fittest feasible individual of the final population, or the fittest overall (infeasible)
                when none is feasible.
        """
        bound = lower_bound_l2(instance)
        bin_limit = math.ceil(BIN_LIMIT_SLACK * bound)
        rate = self.mutation_rate if self.mutation_rate is not None else 1 / instance.get_num_items()
        initial = [create_random_individual(instance, bin_limit, rng) for _ in range(self.population_size)]
        population = rank([self.evaluate(chromosome, instance, bin_limit) for chromosome in initial])
        self.history = []
        for _ in range(self.generations):
            if self.reaches_bound(population, instance.get_capacity(), bound):
                break
            children = self.breed(population, bin_limit, rate, rng)
            evaluated = [self.evaluate(child, instance, bin_limit) for child in children]
            population = rank(population + evaluated)[: self.population_size]
            self.history.append(
                (population[0].fitness, population[len(population) // 2].fitness, population[-1].fitness)
            )
        return self.decode(self.pick_result(population, instance.get_capacity()), instance)

    def breed(self, population: list[Individual], bin_limit: int, rate: float, rng: Generator) -> list[list[int]]:
        """Creates the children of one generation, pairing consecutive selected parents: (p0, p1), (p1, p2), ...

        Args:
            population (list[Individual]): Current population.
            bin_limit (int): Number of bins a chromosome may use.
            rate (float): Mutation probability per gene.
            rng (Generator): Source of randomness.

        Returns:
            list[list[int]]: offspring_per_generation mutated children.
        """
        fitnesses = [individual.fitness for individual in population]
        parents = self.select(fitnesses, self.offspring_per_generation + 1, rng)
        children = []
        for first, second in zip(parents, parents[1:], strict=False):
            child = self.cross(population[first].chromosome, population[second].chromosome, rng)
            children.append(self.mutate(child, bin_limit, rate, rng))
        return children

    def evaluate(self, chromosome: list[int], instance: Instance, bin_limit: int) -> Individual:
        """Returns: Individual: The chromosome with its bin loads and penalized fitness."""
        loads = compute_loads(chromosome, instance, bin_limit)
        fitness = compute_penalized_fitness(loads, instance.get_capacity(), self.penalty_severity)
        return Individual(chromosome, loads, fitness)

    def reaches_bound(self, population: list[Individual], capacity: int, bound: int) -> bool:
        """Returns: bool: True if a feasible individual uses as few bins as the lower bound, proving it optimal."""
        return any(
            individual.is_feasible(capacity) and individual.get_num_used_bins() == bound for individual in population
        )

    def pick_result(self, population: list[Individual], capacity: int) -> Individual:
        """Returns: Individual: The fittest feasible individual, or the fittest overall when none is feasible."""
        feasible = [individual for individual in population if individual.is_feasible(capacity)]
        return feasible[0] if feasible else population[0]

    def decode(self, individual: Individual, instance: Instance) -> Solution:
        """Returns: Solution: The packing the chromosome describes, without its empty bins."""
        bins: list[list[int]] = [[] for _ in individual.loads]
        for item, bin_index in enumerate(individual.chromosome):
            bins[bin_index].append(item)
        return Solution(instance, [items for items in bins if items])

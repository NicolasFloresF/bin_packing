# Bin Packing

Solving the classic one-dimensional bin packing problem with a genetic algorithm and comparing it against reference heuristics on generated instances.

## Language

### Problem

**Item**:
An object with a positive integer size that must be placed in exactly one bin.
_Avoid_: Object, element, weight

**Capacity**:
The maximum total item size a bin may hold; the same for every bin of an instance.
_Avoid_: Bin size, limit

**Bin**:
A container of the instance's capacity; bins are unlimited in number and identical.
_Avoid_: Container, box

**Instance**:
A list of item sizes together with one capacity.
_Avoid_: Problem, dataset, test case

**Solution**:
A packing of every item of an instance into bins; it is feasible when no bin exceeds the capacity.
_Avoid_: Answer, result, packing

**Lower Bound**:
A number of bins that no feasible solution of an instance can go below.
_Avoid_: Optimum, LB (in prose)

### Instance generation

**Size Scaling**:
How generated item sizes relate to the capacity: **scaled** sizes grow with the capacity so every capacity has the same relative difficulty; **unscaled** sizes come from one fixed distribution, so larger capacities make packing easier.
_Avoid_: Design 1, Design 2

### Solving

**Solver**:
A method that receives an instance and returns a solution.
_Avoid_: Algorithm, heuristic (when meaning any solver)

**Heuristic**:
A constructive solver that packs the items in one pass, largest first, following a bin selection rule (e.g. First Fit Decreasing).
_Avoid_: Baseline, greedy

**Bin Selection Rule**:
How a heuristic chooses, among the open bins where an item fits, the one that receives it: **First Fit** (the earliest opened), **Best Fit** (the fullest), **Worst Fit** (the emptiest) or **Random Fit** (one drawn at random); a new bin opens when none fits.
_Avoid_: Strategy, policy

**Genetic Algorithm**:
A solver that evolves a population of candidate solutions through selection, crossover and mutation.
_Avoid_: GA (in prose), evolutionary solver

**Encoding**:
How a chromosome represents a solution; each genetic algorithm here has its own.
_Avoid_: Representation, chromosome type

**Genetic Operator**:
One interchangeable step of a genetic algorithm: initialization, selection, crossover or mutation.
_Avoid_: Strategy, component

**Assignment GA**:
The genetic algorithm whose chromosome gives, for each item, the bin it goes to; candidates may overload bins.
_Avoid_: Original GA, direct GA

**Permutation GA**:
The genetic algorithm whose chromosome is an order of the items, turned into a packing by First Fit; candidates are always feasible.
_Avoid_: Order GA, indirect GA

**Bin Limit**:
The fixed number of bins the Assignment GA may assign items to in an instance.
_Avoid_: Max bins, number of bins

**Simulated Annealing**:
A solver that improves a single solution by random neighbour moves, accepting worse ones with a probability that falls as a temperature cools.
_Avoid_: SA (in prose), annealing

**Fitness**:
The score that ranks candidate solutions in the genetic algorithms and the simulated annealing; the mean squared fill ratio of the used bins (Falkenauer), reduced by an overload penalty in the Assignment GA.
_Avoid_: Score, quality

### Experiments

**Solver Configuration**:
A solver together with its parameter values, identified by a unique name (e.g. "Permutation GA, FFD-seeded").
_Avoid_: Variant, setup

**Run**:
One solver configuration solving one instance with one random seed.
_Avoid_: Execution, trial

**Experiment**:
A set of solver configurations, each run on every instance of an instance set with several seeds.
_Avoid_: Benchmark, study

**Gap**:
How far a run's number of bins is above the instance's lower bound, relative to that bound.
_Avoid_: Error, deviation

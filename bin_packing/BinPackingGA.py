import random
import numpy as np
import matplotlib.pyplot as plt


def random_sampling_generator(items: list, binsCapacity: list, sampleSize: int):
    """Generates a random sample of solutions for the bin packing problem

    Args:
        items (list): List of items to be packed
        binsCapacity (list): List of bins capacity
        sampleSize (int): Number of solutions to be generated

    Returns:
        list: List of solutions generated and their respective bins current capacity
    """
    sample = []
    sampleBinsCurCapacity = []

    # iterate over the sample size
    for _ in range(sampleSize):
        solution = []
        binsCurCapacity = [0] * len(binsCapacity)
        for j in range(len(items)):
            # try to find a bin that can hold the item 100 times
            for _ in range(100):
                randNum = random.randint(0, len(binsCapacity) - 1)
                if binsCurCapacity[randNum] + items[j] > binsCapacity[randNum]:
                    randNum = random.randint(0, len(binsCapacity) - 1)
                else:
                    break

            # append the bin number to the solution
            solution.append(randNum)
            binsCurCapacity[solution[j]] += items[j]

        # append the solution and the bins current capacity to the sample
        sampleBinsCurCapacity.append(binsCurCapacity)
        sample.append(solution)

    return sample, sampleBinsCurCapacity


def overloadPenalty(binsCurCapacity: list, binsCapacity: list):
    """Calculates the overload penalty for the current bins capacity

    Args:
        binsCurCapacity (list): List of bins current capacity
        binsCapacity (list): List of bins capacity

    Returns:
        int: Overload penalty value
    """
    overload = 0
    for i in range(len(binsCapacity)):
        overPercentage = binsCurCapacity[i] / binsCapacity[i]
        if overPercentage <= 1:
            overPercentage = 0
        overload += max(0, overPercentage)
    return overload


def fitness_function(binsCurCapacity: list, binsCapacity: list):
    """Calculates the fitness value for a solution

    Args:
        binsCurCapacity (list): List of bins current capacity
        binsCapacity (list): List of bins capacity

    Returns:
        float: Fitness value
    """
    k = 2
    sum = 0
    bins = 0
    for i in range(len(binsCapacity)):
        if binsCurCapacity[i] > 0:
            bins += 1
        sum += (binsCurCapacity[i] / binsCapacity[i]) ** k
    fitness = sum / max(1, bins)  # len(binsCapacity)

    # put the max fitness value to 1 (invalid solution can have a fitness value greater than 1)
    fitness = min(1, fitness)

    # apply the overload penalty
    fitness /= max(1, overloadPenalty(binsCurCapacity, binsCapacity))

    return fitness


def pointCrossOver(parent1: list, parent2: list, binsCapacity: list, items: list):
    """Performs a point crossover between two parents

    Args:
        parent1 (list): First parent solution
        parent2 (list): Second parent solution
        binsCapacity (list): List of bins capacity
        items (list): List of items to be packed

    Returns:
        list, list: Child solution and bins current capacity
    """
    binsCurCapacity = [0] * len(binsCapacity)
    child = []
    for i in range(len(parent1)):
        if i < len(parent1) / 2:
            child.append(parent1[i])
            binsCurCapacity[parent1[i]] += items[i]
        else:
            child.append(parent2[i])
            binsCurCapacity[parent2[i]] += items[i]

    return child, binsCurCapacity


def UniformCrossOver(parent1: list, parent2: list, binsCapacity: list, items: list):
    """Performs a uniform crossover between two parents

    Args:
        parent1 (list): First parent solution
        parent2 (list): Second parent solution
        binsCapacity (list): List of bins capacity
        items (list): List of items to be packed

    Returns:
        list, list: Child solution and bins current capacity
    """
    binsCurCapacity = [0] * len(binsCapacity)
    child = []
    for i in range(len(parent1)):
        randomNum = random.random()

        if randomNum < 0.5:
            child.append(parent1[i])
            binsCurCapacity[parent1[i]] += items[i]

        if randomNum > 0.5:
            child.append(parent2[i])
            binsCurCapacity[parent2[i]] += items[i]

    return child, binsCurCapacity


def singlePointMutation(child: list, binsCapacity: list, items: list, binsCurCapacity: list):
    """Performs a single point mutation in the child solution

    Args:
        child (list): Child solution
        binsCapacity (list): List of bins capacity
        items (list): List of items to be packed
        binsCurCapacity (list): List of bins current capacity

    Returns:
        list, list: Mutated child solution and bins current capacity
    """
    for i in range(len(child)):
        if random.random() < 0.2:
            randNum = random.randint(0, len(binsCapacity) - 1)
            binsCurCapacity[child[i]] -= items[i]
            binsCurCapacity[randNum] += items[i]
            child[i] = randNum

    return child, binsCurCapacity


def tournamentSelection(sample: list, fitness_list: list):
    """Performs a tournament selection to choose the parents for the next generation

    Args:
        sample (list): List of solutions
        fitness_list (list): List of fitness values for each solution

    Returns:
        list: List of chosen parents
    """

    # calculate the probability of each solution to be chosen
    probability = np.asarray(fitness_list).astype("float64")
    probability /= np.sum(probability)

    # print("Chosen solutions:")
    parents = np.random.choice(len(sample), 50, p=probability)

    return parents


def generateRandomProblem(n, m):
    """Generates a random problem for the bin packing problem

    Args:
        n (int): number of items to be generated
        m (int): number of bins to be generated

    Returns:
        list, list: List of items and list of bins capacity
    """
    items = []
    for i in range(n):
        items.append(random.randint(1, 10))
    binsCapacity = []
    for i in range(m):
        binsCapacity.append(random.randint(10, 20))
    return items, binsCapacity


def main():
    items = [7, 5, 1, 5, 3, 4, 5, 10, 8, 3, 2, 5, 1, 4, 3, 7, 4]
    binsCapacity = [10, 10, 15, 10, 15, 12, 10, 10, 10]

    random.seed(1)
    sample, sampleBinsCurCapacity = random_sampling_generator(items, binsCapacity, 50)

    fitness_list = []
    for i in range(len(sampleBinsCurCapacity)):
        fitness_list.append(fitness_function(sampleBinsCurCapacity[i], binsCapacity))

    x = []
    bestFit = []
    worstFit = []
    medianFit = []
    for gen in range(5000):
        if fitness_list[0] == 1:
            break

        parents = tournamentSelection(sample, fitness_list)

        for i in range(int(len(parents) / 2)):
            child, bin = pointCrossOver(sample[parents[i]], sample[parents[i + 1]], binsCapacity, items)
            child, bin = singlePointMutation(child, binsCapacity, items, bin)
            childFit = fitness_function(bin, binsCapacity)
            sample.append(child)
            sampleBinsCurCapacity.append(bin)
            fitness_list.append(childFit)

        zipped_lists = zip(fitness_list, sample, sampleBinsCurCapacity)
        sorted_zipped_lists = sorted(zipped_lists, reverse=True)
        fitness_list, sample, sampleBinsCurCapacity = [list(t) for t in zip(*sorted_zipped_lists)]

        while len(sample) > 50:
            sample.pop()
            fitness_list.pop()
            sampleBinsCurCapacity.pop()

        print(
            gen,
            fitness_list[0],
            sample[0],
            sampleBinsCurCapacity[0],
            overloadPenalty(sampleBinsCurCapacity[0], binsCapacity),
            fitness_list[-1],
            sample[-1],
            sampleBinsCurCapacity[-1],
            overloadPenalty(sampleBinsCurCapacity[-1], binsCapacity),
        )

        x.append(gen)
        bestFit.append(fitness_list[0])
        medianFit.append(fitness_list[int(len(fitness_list) / 2)])
        worstFit.append(fitness_list[-1])

    plt.plot(x, bestFit, label="Best Fitness")
    plt.plot(x, medianFit, label="Median Fitness")
    plt.plot(x, worstFit, label="Worst Fitness")
    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.show()


if __name__ == "__main__":
    main()

import random
import numpy as np


def checkCurCapacity(items: list, binsCapacity: list, solution: list):
    binsCurCapacity = [0] * len(binsCapacity)

    for i in range(len(items)):
        sum = binsCurCapacity[solution[i]] + items[i]
        if sum > binsCapacity[solution[i]]:
            print(sum, binsCapacity[solution[i]], i)
            print("Invalid solution")
            break
        else:
            binsCurCapacity[solution[i]] = sum

    return binsCurCapacity


def first_fit(items, binsCapacity):
    binsCurCapacity = [0] * len(binsCapacity)
    solution = []
    for i in range(len(items)):
        j = 0
        while j < len(binsCapacity):
            if binsCurCapacity[j] + items[i] <= binsCapacity[j]:
                binsCurCapacity[j] += items[i]
                solution.append(j)
                break
            j += 1
        if j == len(binsCapacity):
            print("Invalid solution")
            break
    return solution, binsCurCapacity


def random_sampling_generator(items, binsCapacity, sampleSize):
    sample = []
    sampleBinsCurCapacity = []
    for _ in range(sampleSize):
        solution = []
        binsCurCapacity = [0] * len(binsCapacity)
        for j in range(len(items)):
            for _ in range(100):
                randNum = random.randint(0, len(binsCapacity) - 1)
                if binsCurCapacity[randNum] + items[j] > binsCapacity[randNum]:
                    randNum = random.randint(0, len(binsCapacity) - 1)
                else:
                    break
            solution.append(randNum)
            binsCurCapacity[solution[j]] += items[j]

        sampleBinsCurCapacity.append(binsCurCapacity)
        sample.append(solution)
    return sample, sampleBinsCurCapacity


def overloadPenalty(binsCurCapacity: list, binsCapacity: list):
    overload = 0
    for i in range(len(binsCapacity)):
        overPercentage = binsCurCapacity[i] / binsCapacity[i]
        if overPercentage <= 1:
            overPercentage = 0
        # print(overPercentage)
        overload += max(0, overPercentage)
    return overload


def fitness_function(binsCurCapacity: list, binsCapacity: list):
    # The Fitness Function implementation is based on the following article https://www.codeproject.com/Articles/633133/Genetic-Algorithm-for-Bin-Packing-Problem
    # F = Σⁿᵢ[(fᵢ/cᵢ)ᵏ]/n
    # Where:
    # F - fitness of the solution
    # n - number of bins
    # fi - fill of the ith bin
    # c - capacity of the bin
    # k - constant greater then 1
    # constant controls whether the more filled bins are preferred or equally filled bins. Larger values should be used in case more filled bins are preferred. This example sets value of k to 2.
    k = 3
    sum = 0
    bins = 0
    for i in range(len(binsCapacity)):
        if binsCurCapacity[i] > 0:
            bins += 1
        sum += (binsCurCapacity[i] / binsCapacity[i]) ** k
    fitness = sum / max(1, bins)  # len(binsCapacity)
    # 0   0  5  4 13  7  4 10  5
    # 10 10 15 10 15 12 10 10 10
    # 0/7
    # 0 - 1
    # -1
    # 1
    # 0 0 0 0 0 0 8 10 12
    #! implent overload penalty
    fitness = min(1, fitness)
    fitness /= max(1, overloadPenalty(binsCurCapacity, binsCapacity))
    # print(binsCurCapacity, overloadPenalty(binsCurCapacity, binsCapacity))
    return fitness


def UniformCrossOver(parent1, parent2, binsCapacity, items):
    binsCurCapacity = [0] * len(binsCapacity)
    child = []
    for i in range(len(parent1)):
        # flag = False
        randomNum = random.random()

        # try to add the item to the bin of the first parent
        # if binsCurCapacity[parent1[i]] + items[i] <= binsCapacity[parent1[i]]:
        #     child.append(parent1[i])
        #     binsCurCapacity[parent1[i]] += items[i]
        # elif randomNum < 0.5 and binsCurCapacity[parent2[i]] + items[i] <= binsCapacity[parent2[i]]:
        #     child.append(parent2[i])
        #     binsCurCapacity[parent2[i]] += items[i]
        # else:
        #     flag = True

        if randomNum < 0.5:
            child.append(parent1[i])
            binsCurCapacity[parent1[i]] += items[i]

        # try to add the item to the bin of the second parent
        if randomNum > 0.5:
            child.append(parent2[i])
            binsCurCapacity[parent2[i]] += items[i]

    return child, binsCurCapacity


def mutation(child, binsCapacity, items, binsCurCapacity):
    for i in range(len(child)):
        if random.random() < 0.2:
            for _ in range(100):
                randNum = random.randint(0, len(binsCapacity) - 1)
                if binsCurCapacity[randNum] + items[i] > binsCapacity[randNum]:
                    randNum = random.randint(0, len(binsCapacity) - 1)
                else:
                    binsCurCapacity[child[i]] -= items[i]
                    binsCurCapacity[randNum] += items[i]
                    child[i] = randNum
                    break

    # checkCurCapacity(items, binsCapacity, child)
    return child, binsCurCapacity


def selection(sample, fitness_list):
    # calculate the probability of each solution to be chosen
    p = np.asarray(fitness_list).astype("float64")
    p /= np.sum(p)

    # print(p)

    # print("Chosen solutions:")
    parents = np.random.choice(len(sample), 50, p=p)

    return parents


def main():
    # inputs
    # items = [5, 5, 4, 7, 1, 3]
    items = [7, 5, 1, 5, 3, 4, 5, 10, 8]
    binsCapacity = [10, 10, 15, 10, 15, 12, 10, 10, 10]

    # output
    # solution = [0, 1, 2, 2, 0, 1]

    # binsCurCapacity = checkCurCapacity(items, binsCapacity, solution)

    # print(solution, binsCurCapacity)
    # print(fitness_function(binsCurCapacity, binsCapacity))
    #
    # solution, binsCurCapacity = first_fit(items, binsCapacity)
    #
    # print(solution, binsCurCapacity)
    # print(fitness_function(binsCurCapacity, binsCapacity))

    #! invalid solution
    #! duplicated or similar solutions (heuristic solutions? - first fit)
    random.seed(1)
    sample, sampleBinsCurCapacity = random_sampling_generator(items, binsCapacity, 50)

    fitness_list = []
    for i in range(len(sampleBinsCurCapacity)):
        fitness_list.append(fitness_function(sampleBinsCurCapacity[i], binsCapacity))

    # for i in range(len(sample)):
    #     print(fitness_list[i], sample[i], sampleBinsCurCapacity[i])

    # print("-------------------------------------")

    for gen in range(1000):
        parents = selection(sample, fitness_list)

        for i in range(int(len(parents) / 2)):
            child, bin = UniformCrossOver(sample[parents[i]], sample[parents[i + 1]], binsCapacity, items)
            child, bin = mutation(child, binsCapacity, items, bin)
            childFit = fitness_function(bin, binsCapacity)
            # print(child, bin, childFit)
            sample.append(child)
            sampleBinsCurCapacity.append(bin)
            fitness_list.append(childFit)

        # child, bin = UniformCrossOver(sample[parents[0]], sample[parents[1]], binsCapacity, items)
        # print("uniform crossover")
        # print(child, bin)
        # print(fitness_function(bin, binsCapacity))
        # if child is not None:
        #     child, bin = mutation(child, binsCapacity, items, bin)
        #     print("mutation")
        #     print(child, bin)
        #     childFit = fitness_function(bin, binsCapacity)
        #     print(childFit)

        # fitness_list.append(childFit)
        # sample.append(child)
        # sampleBinsCurCapacity.append(bin)

        # print(len(fitness_list), len(sample), len(sampleBinsCurCapacity))

        # # zip the fitness values with the sample and sort them in descending order
        # #!important: this is the right zip order: fitness_list, sample, sampleBinsCurCapacity

        zipped_lists = zip(fitness_list, sample, sampleBinsCurCapacity)
        sorted_zipped_lists = sorted(zipped_lists, reverse=True)
        # fitness_list, sample, sampleBinsCurCapacity = zip(*sorted_zipped_lists)
        # unzip with list comprehension
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
            fitness_list[-1],
            sample[-1],
            sampleBinsCurCapacity[-1],
        )

    # print(len(fitness_list), len(sample), len(sampleBinsCurCapacity))

    # print the fitness values for each solution in the sample
    # for i in range(len(sample)):
    #     print(fitness_list[i], sample[i], sampleBinsCurCapacity[i])

    # print(fitness_list)
    # print(sample)
    # print(sampleBinsCurCapacity)


if __name__ == "__main__":
    main()

import random


def checkCurCapacity(items: list, binsCapacity: list, solution: list):
    binsCurCapacity = [0] * len(binsCapacity)

    for i in range(len(items)):
        sum = binsCurCapacity[solution[i]] + items[i]
        if sum > binsCapacity[solution[i]]:
            print(sum)
            print("Invalid solution")
            break
        else:
            binsCurCapacity[solution[i]] = sum

    return binsCurCapacity


def random_sampling_generator(items, binsCapacity, sampleSize):
    sample = []
    sampleBinsCurCapacity = []
    for _ in range(sampleSize):
        solution = []
        binsCurCapacity = [0] * len(binsCapacity)
        for j in range(len(items)):
            while True:
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
    k = 2
    sum = 0
    for i in range(len(binsCapacity)):
        sum += (binsCurCapacity[i] / binsCapacity[i]) ** k
    fitness = sum / len(binsCapacity)
    return fitness


def genetic_algorithm(sample, sampleBinsCurCapacity, binsCapacity):
    fitness_list = []
    for i in range(len(sampleBinsCurCapacity)):
        fitness_list.append(fitness_function(sampleBinsCurCapacity[i], binsCapacity))
    # O sort faz perder a ordem dos items
    # fitness_list.sort(reverse=True)
    # sort all trhee lists zipped
    for i in range(len(sample)):
        print(fitness_list[i], sample[i], sampleBinsCurCapacity[i])
    print("-------------------------------------")
    zipped_lists = zip(fitness_list, sample, sampleBinsCurCapacity)
    sorted_zipped_lists = sorted(zipped_lists, reverse=True)
    fitness_list, sample, sampleBinsCurCapacity = zip(*sorted_zipped_lists)

    for i in range(len(sample)):
        print(fitness_list[i], sample[i], sampleBinsCurCapacity[i])
    # print(fitness_list)


def main():
    # inputs
    # items = [5, 5, 4, 7, 1, 3]
    items = [7, 5, 1, 5, 3, 4]
    binsCapacity = [10, 10, 15]

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

    sample, sampleBinsCurCapacity = random_sampling_generator(items, binsCapacity, 100)

    genetic_algorithm(sample, sampleBinsCurCapacity, binsCapacity)
    # print(sample)
    # print(sampleBinsCurCapacity)


if __name__ == "__main__":
    main()

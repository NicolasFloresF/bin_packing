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


def genetic_algorithm(): ...


def main():
    # inputs
    items = [5, 5, 4, 7, 1, 3]
    binsCapacity = [10, 10, 15]

    # output
    solution = [0, 1, 2, 2, 0, 1]
    binsCurCapacity = [0, 0, 0]

    for i in range(len(items)):
        sum = binsCurCapacity[solution[i]] + items[i]
        if sum > binsCapacity[solution[i]]:
            print(sum)
            print("Invalid solution")
            break
        else:
            binsCurCapacity[solution[i]] = sum

    print(binsCurCapacity)
    print(fitness_function(binsCurCapacity, binsCapacity))


if __name__ == "__main__":
    main()

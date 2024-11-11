# First fit for bin packing problem


def sort_descending(weights):
    # Sort the weights in descending order
    weights.sort(reverse=True)
    return weights


def first_fit(weights, capacity):
    # Number of bins required
    bins = list()

    # Loop through each item
    for i in range(len(weights)):
        j = 0
        while j < len(bins):
            if bins[j] + weights[i] <= capacity:
                bins[j] += weights[i]
                break
            j += 1
        if j == len(bins):
            bins.append(weights[i])

    # Print the number of bins required
    print("Number of bins required in First Fit : ", len(bins))
    print("Bins are : ", bins)


def main():
    # List of weights of items
    weights = [5, 5, 4, 7, 1, 3, 8]

    # Capacity of bins
    capacity = 10

    # Sort the weights in descending order
    # weights = sort_descending(weights)

    # Function call
    first_fit(weights, capacity)


if __name__ == "__main__":
    main()

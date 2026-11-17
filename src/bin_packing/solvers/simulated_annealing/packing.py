from bin_packing.instance.instance import Instance


class Packing:
    """Holds a feasible packing that changes by moving or swapping items, keeping its Falkenauer fitness up to date.

    Bins keep their index for the whole search; a bin emptied by a move stays in the list as an empty, unused bin.
    """

    def __init__(self, instance: Instance, bins: list[list[int]]) -> None:
        """Builds the packing from a feasible list of bins.

        Args:
            instance (Instance): Instance the items belong to.
            bins (list[list[int]]): Item indices held by each bin.

        Raises:
            ValueError: If a bin holds more than the capacity.
        """
        self.sizes = instance.get_item_sizes()
        self.capacity = instance.get_capacity()
        self.bins = [list(items) for items in bins]
        self.loads = [sum(self.sizes[item] for item in items) for items in self.bins]
        for bin_index, load in enumerate(self.loads):
            if load > self.capacity:
                raise ValueError(f"Bin {bin_index} holds {load}, above the capacity {self.capacity}.")
        self.bin_of_item = [0] * len(self.sizes)
        for bin_index, items in enumerate(self.bins):
            for item in items:
                self.bin_of_item[item] = bin_index
        self.sum_of_squared_loads = sum(load**2 for load in self.loads)
        self.num_used_bins = sum(1 for load in self.loads if load > 0)

    def get_fitness(self) -> float:
        """Returns: float: The Falkenauer fitness of the packing (mean squared fill ratio of the used bins)."""
        return self.fitness_of(self.sum_of_squared_loads, self.num_used_bins)

    def get_num_used_bins(self) -> int:
        """Returns: int: Number of bins holding at least one item."""
        return self.num_used_bins

    def get_num_bins(self) -> int:
        """Returns: int: Number of bin slots, empty ones included."""
        return len(self.bins)

    def get_num_items(self) -> int:
        """Returns: int: Number of items packed."""
        return len(self.sizes)

    def get_bin_of_item(self, item: int) -> int:
        """Returns: int: Index of the bin holding the item."""
        return self.bin_of_item[item]

    def get_used_bins(self) -> list[list[int]]:
        """Returns: list[list[int]]: A copy of the item indices of each non-empty bin."""
        return [list(items) for items in self.bins if items]

    def fitness_of(self, sum_of_squared_loads: int, num_used_bins: int) -> float:
        """Returns: float: The Falkenauer fitness for the given sum of squared loads and number of used bins."""
        return sum_of_squared_loads / (self.capacity**2 * num_used_bins)

    def fitness_after_move(self, item: int, target: int) -> float | None:
        """Computes the fitness the packing would have after moving an item to another bin, without moving it.

        Args:
            item (int): Item to move.
            target (int): Bin that would receive it.

        Returns:
            float | None: The new fitness, or None if the target is the item's bin, is empty, or lacks room.
        """
        source = self.bin_of_item[item]
        size = self.sizes[item]
        if target == source or self.loads[target] == 0 or self.loads[target] + size > self.capacity:
            return None
        old_source, old_target = self.loads[source], self.loads[target]
        new_source, new_target = old_source - size, old_target + size
        squares = self.sum_of_squared_loads - old_source**2 - old_target**2 + new_source**2 + new_target**2
        return self.fitness_of(squares, self.num_used_bins - (1 if new_source == 0 else 0))

    def apply_move(self, item: int, target: int) -> None:
        """Moves an item to another bin; the caller has checked the move with fitness_after_move."""
        source = self.bin_of_item[item]
        size = self.sizes[item]
        self.sum_of_squared_loads -= self.loads[source] ** 2 + self.loads[target] ** 2
        self.bins[source].remove(item)
        self.bins[target].append(item)
        self.loads[source] -= size
        self.loads[target] += size
        self.sum_of_squared_loads += self.loads[source] ** 2 + self.loads[target] ** 2
        self.bin_of_item[item] = target
        if self.loads[source] == 0:
            self.num_used_bins -= 1

    def fitness_after_swap(self, item_a: int, item_b: int) -> float | None:
        """Computes the fitness the packing would have after exchanging two items' bins, without exchanging them.

        Args:
            item_a (int): First item.
            item_b (int): Second item.

        Returns:
            float | None: The new fitness, or None if both items share a bin or either bin would overflow.
        """
        bin_a, bin_b = self.bin_of_item[item_a], self.bin_of_item[item_b]
        difference = self.sizes[item_b] - self.sizes[item_a]
        if bin_a == bin_b:
            return None
        new_a, new_b = self.loads[bin_a] + difference, self.loads[bin_b] - difference
        if new_a > self.capacity or new_b > self.capacity:
            return None
        squares = self.sum_of_squared_loads - self.loads[bin_a] ** 2 - self.loads[bin_b] ** 2 + new_a**2 + new_b**2
        return self.fitness_of(squares, self.num_used_bins)

    def apply_swap(self, item_a: int, item_b: int) -> None:
        """Exchanges two items' bins; the caller has checked the swap with fitness_after_swap."""
        bin_a, bin_b = self.bin_of_item[item_a], self.bin_of_item[item_b]
        difference = self.sizes[item_b] - self.sizes[item_a]
        self.sum_of_squared_loads -= self.loads[bin_a] ** 2 + self.loads[bin_b] ** 2
        self.bins[bin_a].remove(item_a)
        self.bins[bin_b].remove(item_b)
        self.bins[bin_a].append(item_b)
        self.bins[bin_b].append(item_a)
        self.loads[bin_a] += difference
        self.loads[bin_b] -= difference
        self.sum_of_squared_loads += self.loads[bin_a] ** 2 + self.loads[bin_b] ** 2
        self.bin_of_item[item_a], self.bin_of_item[item_b] = bin_b, bin_a

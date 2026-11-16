from bin_packing.instance.instance import Instance


class Solution:
    """Holds a packing of every item of an instance into bins; the packing may overload bins."""

    def __init__(self, instance: Instance, bins: list[list[int]]) -> None:
        """Builds a solution after checking that every item sits in exactly one non-empty bin.

        Args:
            instance (Instance): Instance the packing belongs to.
            bins (list[list[int]]): Item indices held by each bin.

        Raises:
            ValueError: If a bin is empty, an index is out of range, or an item is missing or repeated.
        """
        seen = [False] * instance.get_num_items()
        for bin_index, items in enumerate(bins):
            if not items:
                raise ValueError(f"Bin {bin_index} of the solution for {instance.get_name()!r} is empty.")
            for item in items:
                if not 0 <= item < len(seen):
                    raise ValueError(f"Invalid item index {item!r} in bin {bin_index}. Expected 0 to {len(seen) - 1}.")
                if seen[item]:
                    raise ValueError(f"Item {item} appears more than once in the solution.")
                seen[item] = True
        missing = [item for item, placed in enumerate(seen) if not placed]
        if missing:
            raise ValueError(f"Items {missing} are missing from the solution for {instance.get_name()!r}.")
        self.instance = instance
        self.bins = [list(items) for items in bins]

    def get_instance(self) -> Instance:
        """Returns: Instance: Instance the packing belongs to."""
        return self.instance

    def get_bins(self) -> list[list[int]]:
        """Returns: list[list[int]]: A copy of the item indices held by each bin."""
        return [list(items) for items in self.bins]

    def get_num_bins(self) -> int:
        """Returns: int: Number of bins used."""
        return len(self.bins)

    def get_bin_loads(self) -> list[int]:
        """Returns: list[int]: Total item size held by each bin."""
        sizes = self.instance.get_item_sizes()
        return [sum(sizes[item] for item in items) for items in self.bins]

    def is_feasible(self) -> bool:
        """Returns: bool: True if no bin holds more than the capacity."""
        capacity = self.instance.get_capacity()
        return all(load <= capacity for load in self.get_bin_loads())

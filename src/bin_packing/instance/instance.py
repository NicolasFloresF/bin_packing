class Instance:
    """Holds the item sizes and the bin capacity of one bin packing instance."""

    def __init__(self, name: str, item_sizes: list[int], capacity: int) -> None:
        """Builds an instance after checking that every item fits in an empty bin.

        Args:
            name (str): Identifier of the instance, used in file names and results.
            item_sizes (list[int]): Size of each item; item indices follow this order.
            capacity (int): Capacity shared by every bin.

        Raises:
            ValueError: If the name is empty, the capacity is not positive, there are no items, or an item size
                is outside [1, capacity].
        """
        if not name:
            raise ValueError(f"Invalid instance name: {name!r}. Expected a non-empty string.")
        if capacity < 1:
            raise ValueError(f"Invalid capacity: {capacity!r}. Expected a positive integer.")
        if not item_sizes:
            raise ValueError(f"Instance {name!r} has no items. Expected at least one.")
        for index, size in enumerate(item_sizes):
            if not 1 <= size <= capacity:
                raise ValueError(f"Invalid size {size!r} for item {index} of {name!r}. Expected 1 to {capacity}.")
        self.name = name
        self.item_sizes = tuple(item_sizes)
        self.capacity = capacity

    def get_name(self) -> str:
        """Returns: str: Identifier of the instance."""
        return self.name

    def get_item_sizes(self) -> tuple[int, ...]:
        """Returns: tuple[int, ...]: Size of each item, indexed by item."""
        return self.item_sizes

    def get_capacity(self) -> int:
        """Returns: int: Capacity shared by every bin."""
        return self.capacity

    def get_num_items(self) -> int:
        """Returns: int: Number of items."""
        return len(self.item_sizes)

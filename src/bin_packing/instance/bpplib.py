from pathlib import Path

from bin_packing.instance.instance import Instance

# Number of itens and bin capacity
HEADER_LENGTH = 2


def write_instance(instance: Instance, directory: Path) -> Path:
    """Writes an instance in the BPPLIB text format: the number of items, the capacity, then one size per line.

    Args:
        instance (Instance): Instance to write.
        directory (Path): Directory that receives the file, created if missing.

    Returns:
        Path: The written file, named after the instance.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{instance.get_name()}.txt"
    lines = [str(instance.get_num_items()), str(instance.get_capacity())]
    lines += [str(size) for size in instance.get_item_sizes()]
    path.write_text("\n".join(lines) + "\n")
    return path


def read_instance(path: Path) -> Instance:
    """Reads an instance in the BPPLIB text format, naming it after the file.

    Args:
        path (Path): File to read.

    Returns:
        Instance: The instance described by the file.

    Raises:
        ValueError: If a line is not an integer or the number of sizes differs from the declared number of items.
    """
    tokens = path.read_text().split()
    try:
        values = [int(token) for token in tokens]
    except ValueError as error:
        raise ValueError(f"Invalid BPPLIB file {str(path)!r}: {error}. Expected integers only.") from error
    if len(values) < HEADER_LENGTH:
        raise ValueError(f"Invalid BPPLIB file {str(path)!r}. Expected the number of items and the capacity.")
    num_items, capacity, sizes = values[0], values[1], values[HEADER_LENGTH:]
    if len(sizes) != num_items:
        raise ValueError(f"Invalid BPPLIB file {str(path)!r}: declares {num_items} items but lists {len(sizes)} sizes.")
    return Instance(path.stem, sizes, capacity)

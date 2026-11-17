import warnings

import numpy as np

from bin_packing.instance.instance import Instance

SIZE_SCALINGS = ("scaled", "unscaled")
WEIBULL_SHAPE = 3.0
# Setting the Weibull scale to 45% of the capacity (scaled) or to a fixed 45 (unscaled), which puts the mean item
# size at about 40% of the capacity, or at about 40 units.
SCALED_SCALE_RATIO = 0.45
UNSCALED_SCALE = 45.0
# Warning above 1% of clipped items: at that point the sizes visibly stop following the Weibull shape.
# Unscaled sizes cross it at capacities below about 75; scaled sizes clip about 0.002% at capacity 150 but can cross
# it at tiny capacities such as 10, where rounding sends small sizes to 0.
CLIPPED_WARNING_RATIO = 0.01


def generate_weibull_instance(num_items: int, capacity: int, size_scaling: str, seed: int) -> Instance:
    """Generates an instance whose item sizes follow a Weibull distribution, rounded and clipped to [1, capacity].

    Args:
        num_items (int): Number of items to generate.
        capacity (int): Capacity shared by every bin.
        size_scaling (str): "scaled" to grow sizes with the capacity, "unscaled" for one fixed distribution.
        seed (int): Seed of the random generator, also written in the instance name.

    Returns:
        Instance: The generated instance, named after its parameters.

    Raises:
        ValueError: If the size scaling is unknown or the number of items is not positive.
    """
    if size_scaling not in SIZE_SCALINGS:
        raise ValueError(f"Invalid size scaling: {size_scaling!r}. Expected one of {SIZE_SCALINGS}.")
    if num_items < 1:
        raise ValueError(f"Invalid number of items: {num_items!r}. Expected a positive integer.")

    scale = capacity * SCALED_SCALE_RATIO if size_scaling == "scaled" else UNSCALED_SCALE
    rng = np.random.default_rng(seed)
    raw_sizes = np.round(rng.weibull(WEIBULL_SHAPE, num_items) * scale).astype(int)

    clipped_ratio = float(np.mean((raw_sizes > capacity) | (raw_sizes < 1)))
    if clipped_ratio > CLIPPED_WARNING_RATIO:
        warnings.warn(
            f"{clipped_ratio:.1%} of the item sizes were clipped to [1, {capacity}]; "
            f"with {size_scaling!r} sizes this capacity distorts the Weibull distribution.",
            stacklevel=2,
        )

    sizes = np.clip(raw_sizes, 1, capacity)
    name = f"weibull_{size_scaling}_n{num_items}_c{capacity}_s{seed}"
    return Instance(name, sizes.tolist(), capacity)

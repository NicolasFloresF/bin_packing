## Bin Packing Optimization Using Genetic Algorithm

This project solves the classic one-dimensional bin packing problem: items of different sizes must be packed into
identical bins of a fixed capacity, using as few bins as possible. It compares genetic algorithms, simulated annealing
and constructive heuristics on generated instances.

> Work in progress: the project is being restructured. The solvers and the command line are not available yet.

### Prerequisites

The project uses [uv](https://docs.astral.sh/uv/) to manage Python and its dependencies.

### Installation

```bash
git clone https://github.com/NicolasFloresF/bin_packing.git
cd bin_packing
uv sync
```

### Quality checks

```bash
make -C code-quality check
```

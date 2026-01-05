# mec_rust

Rust implementation of performance-critical components for the `mec` (minimum entropy coupling) library.

## Installation

To build, run `maturin develop --release` from this directory.

## What's Implemented

- `greedy_mec`: Rust implementation of the greedy approximation algorithm from [Kocaoglu et al. (2017)](https://arxiv.org/abs/1701.08254)
  - Supports both dense (numpy array) and sparse (dictionary) output formats
  - Significantly faster than the pure Python implementation
- `entropy`: Shannon entropy calculation
- `entropy_upper_bounds`: Entropy upper bounds for tree pruning in iterative algorithms
- `get_proportional_rows`: Find proportional rows in a matrix
- `is_distribution`: Fast distribution validation
- `is_deterministic`: Check if a distribution is deterministic

## Usage

This package is automatically used by the main `mec` library when installed. The Python code will automatically fall back to a pure Python implementation if this package is not available.

You can also use it directly:

```python
import numpy as np
from mec_rust import greedy_mec

# Two marginals
x = np.array([0.5, 0.5])
y = np.array([0.5, 0.5])
coupling = greedy_mec(x, y)

# Sparse output
coupling_sparse = greedy_mec(x, y, sparse=True)
```

## Development

The Rust code is in `src/`:
- `lib.rs`: Main module and Python bindings
- `sparse_coupling.rs`: Sparse coupling implementation

Dependencies are managed in `Cargo.toml`.

## Testing

Tests for the Rust implementation are run through the main `mec` package test suite.

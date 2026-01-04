# Rust Migration Guide

This document describes the Rust migration of the `mec` library.

## Architecture

The library now uses a hybrid Python/Rust architecture:

- **Rust (in `rust/`)**: Performance-critical algorithms and utilities
  - `greedy_mec`: Core coupling algorithm
  - `entropy`: Shannon entropy calculation
  - `entropy_upper_bounds`: Entropy upper bounds for pruning
  - `get_proportional_rows`: Find proportional matrix rows
  - `is_distribution`: Distribution validation
  - `is_deterministic`: Deterministic distribution check

- **Python (in `mec/`)**: High-level APIs and iterative algorithms
  - `TIMEC`, `FIMEC`, `ARIMEC`: Iterative coupling algorithms
  - Marginal and posterior classes
  - High-level utility functions
  - Examples

## Building the Rust Extension

The Rust extension (`mec_rust`) must be built before the main `mec` package can use it:

```bash
cd rust
maturin develop --release
```

Or to build a wheel:

```bash
cd rust
maturin build --release
pip install target/wheels/mec_rust-*.whl
```

## Installation

For users:

```bash
pip install mec
```

For developers:

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the Rust extension
cd rust
maturin develop --release

# Install the main package in editable mode
cd ..
pip install -e .
```

## Testing

All existing tests continue to work:

```bash
pytest tests/
```

The tests will automatically use the Rust implementation if available, or fall back to Python.

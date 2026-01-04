# Performance Benchmarks

## Summary

The Rust migration provides significant performance improvements across all core functions and algorithms.

## Individual Function Speedups (Rust vs Python)

Measured with isolated benchmarks on production-sized inputs:

| Function | Rust Time | Python Time | Speedup |
|----------|-----------|-------------|---------|
| `greedy_mec` (size=50) | 0.13 ms | 0.93 ms | **6.96x** |
| `entropy` (1000 states) | 10.6 μs | 24.4 μs | **2.31x** |
| `is_distribution` | 4.1 μs | 24.3 μs | **5.96x** |

## Iterative Algorithm Performance

FIMEC and ARIMEC benefit from the Rust speedups because they call `greedy_mec` and utility functions thousands of times per sample.

### FIMEC/ARIMEC Timings (with Rust)

Using test marginals (depth=2, vocab=4):

- **FIMEC initialization**: 0.26 ms
- **FIMEC sample**: 12.43 ms
- **ARIMEC initialization**: 0.08 ms
- **ARIMEC sample**: 0.49 ms

### Overall Test Suite Performance

The complete test suite (25 tests including TIMEC, FIMEC, ARIMEC, and greedy_mec):

- **Before Rust utilities**: ~62 seconds
- **After Rust utilities**: ~23 seconds
- **Improvement**: **~63% faster**

## Why the Speedups Matter

1. **`greedy_mec`** is called in tight loops during coupling computation - being 7x faster directly accelerates all iterative algorithms

2. **`entropy` and `is_distribution`** are called dozens to hundreds of times per sample in FIMEC/ARIMEC - 2-6x speedups compound quickly

3. **Overall algorithm performance** improves proportionally to how often these functions are called in the inner loops

## Benchmark Details

All benchmarks were run on:
- Platform: Linux
- Python: 3.10.19
- Numpy: 2.2.6
- Rust: Compiled in release mode with optimizations

Test marginals match those used in the unit tests to ensure realistic workloads.

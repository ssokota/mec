# mec

This repository implements algorithms for minimum-entropy coupling, which is the problem of computing a joint distribution with minimum entropy for given marginals.

Minimum-entropy coupling is equivalent to maximum-mutual-information coupling for the two marginal case.

## Basic Information

The algorithms implemented by this repository come in two classes: 
1. Provable approximation algorithms: For coupling distributions with small supports.
2. Heuristic iterative algorithms: For coupling distributions with large supports.

The package implements three main algorithms:
1. `greedy_mec`: Approximation algorithm for coupling N marginals.
2. `FIMEC`: Iterative algorithm for coupling one factored marginal and one autoregressive marginal.
3. `ARIMEC`: Iterative algorithm for coupling two autoregressive marginals.

## Installation

### Basic Installation

Install the package:
```bash
pip install .
```

Install the package with examples:
```bash
pip install ".[examples]"
```

### Optional: Rust Extension for Better Performance

For significantly better performance, you can build and install the optional Rust extension:

1. Install Rust: https://rustup.rs/
2. Install maturin (the Rust-Python build tool):
   ```bash
   pip install maturin
   ```
3. Build and install the Rust extension:
   ```bash
   cd rust
   maturin develop --release
   cd ..
   ```
4. Verify it's working (should print "Using Rust implementation"):
   ```bash
   python -c "from mec_rust import greedy_mec; print('Using Rust implementation')"
   ```

The package automatically uses the Rust implementation when available and falls back to Python otherwise.

## Usage

### Greedy Approximation Algorithm

To import the greedy approximation algorithm, do:
```
from mec import greedy_mec
```
The greedy approximation algorithm has the following interface:
- Inputs:
     - Variable number N of 1-D numpy arrays, where each array is a probability distribution.
     - Keyword argument `sparse` (optional, defaults `False`), specifying whether to return sparse representation.
- Output: Approximate minimum-entropy coupling
     - If `not sparse`, output is N-D numpy array.
     - If `sparse`, output is dictionary with N-tuple keys only for non-zero entries.

### Iterative Algorithms

To import the iterative algorithms, do:
```
from mec import FIMEC, ARIMEC
```

FIMEC and ARIMEC take three arguments at initialization: 
1. Distribution over X.
   - For FIMEC, must follow FactoredMarginal protocol defined in `mec/iterative/marginals.py`.
   - For ARIMEC, must follow AutoRegressiveMarginal protocol defined in  `mec/iterative/marginals.py`.
2. Distribution over Y following AutoRegressiveMarginal protocol defined in  `mec/iterative/marginals.py`.
3. Whether to use merging (optional, defaults `True`). `True` increases IMEC's robustness to bad marginal parameterizations but increases runtime.

FIMEC and ARIMEC have the following interface:
- `evaluate(x, y) -> log p(x, y)`: Evaluates log likelihood of joint distribution.
- `evaluate_y_given_x(y, x) -> log p(y | x)`: Evaluates log likelihood of conditional distribution.
- `evaluate_x_given_y(x, y) -> log p(x | y)`: Evaluates log likelihood of conditional distribution.
- `sample() -> (x, y), log p(x, y)`: Samples from joint distribution.
- `sample_y_given_x(x) -> y, log p(y | x)`: Samples from conditional distribution.
- `sample_x_given_y(y) -> x, log p(x | y)`: Samples from conditional distribution.
- `estimate_y_given_x(x) -> y, log p(y | x)`: Greedy approximation of maximum a posteriori.
- `estimate_x_given_y(y) -> x, log p(x | y)`: Greedy approximation of maximum a posteriori.

Both `x` and `y` are lists of integers.

## Examples

For examples, see:
- `examples/greedy_mec_example.py` for the greedy algorithm.
- `examples/greedy_mec_sparse_example.py` for the greedy algorithm with sparsity.
- `examples/causal_inference_example.py` for entropic causal inference.
- `examples/fimec_example.py` for FIMEC.
- `examples/arimec_example.py` for ARIMEC.
- `examples/stego_encrypted_example.py` for encrypted steganography.
- `examples/stego_unencrypted_example.py` for unencrypted steganography.

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

  
## Additional Information

### Greedy Approximation Algorithm

The package implements the greedy approximation algorithm from [Kocaoglu et al. (2017)](https://arxiv.org/abs/1701.08254). At the time of writing, this greedy approximation algorithm possesses the best approximation guarantee of existing approaches, per the results of [Compton et al. (2023)](https://arxiv.org/abs/2302.11838). Use cases for approximation algorithms include causal inference, random number generation, multimodal learning, and dimensionality reduction. See [Compton et al. (2023)](https://arxiv.org/abs/2302.11838) for more information about approximation algorithms and their applications.

### Iterative Algorithms

The package implements three iterative algorithms: tabular iterative minimum-entropy coupling (TIMEC) and factored iterative minimum-entropy coupling (FIMEC) from [Sokota et al. (2022)](https://arxiv.org/abs/2107.08295), and autoregressive iterative minimum-entropy coupling (ARIMEC) from [Sokota et al. (2024)](https://arxiv.org/abs/2405.19540). However, FIMEC and ARIMEC generalize TIMEC, so there are functionally only two algorithms. At the time of writing, FIMEC and ARIMEC are the only approaches for computing low-entropy couplings for large-support distributions, such as language models. Use cases for iterative algorithms include communication and steganography. See [Sokota et al. (2024)](https://arxiv.org/abs/2405.19540) for more information about iterative algorithms and [Schroeder de Witt et al. (2023)](https://arxiv.org/abs/2210.14889) for more information about their application to steganography.

## References

ARIMEC, merging, MEC for unencrypted steganography
```
@inproceedings{clec24,
    title = {Computing Low-Entropy Couplings for Large-Support Distributions},
    author = {Samuel Sokota and Dylan Sam and Christian Schroeder de Witt and Spencer Compton and Jakob Foerster and J. Zico Kolter},
    booktitle = {Conference on Uncertainty in Artificial Intelligence (UAI)},
    year = {2024},
}
```

MEC for encrypted steganography
```
@InProceedings{pss23,
    title = {Perfectly Secure Steganography Using Minimum Entropy Coupling},
    author = {Christian Schroeder de Witt and Samuel Sokota and J. Zico Kolter and Jakob Foerster and Martin Strohmeier},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year = {2023},
}
```

IMEC, TIMEC, FIMEC
```
@InProceedings{cmdp22,
    title = {Communicating via {M}arkov Decision Processes},
    author = {Samuel Sokota and Christian Schroeder De Witt and Maximilian Igl and Luisa Zintgraf and Philip Torr and Martin Strohmeier and J. Zico Kolter  and Shimon Whiteson and Jakob Foerster},
    booktitle = {International Conference on Machine Learning (ICML)},
    year = {2022},
}
```

Greedy approximation algorithm, MEC for causal inference
```
@article{eci17, 
    title={Entropic Causal Inference}, 
    journal={AAAI Conference on Artificial Intelligence (AAAI)}, 
    author={Murat Kocaoglu and Alexandros Dimakis and Sriram Vishwanath and Babak Hassibi}, 
    year={2017}
}
```

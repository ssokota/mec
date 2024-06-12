from typing import Literal, overload

import numpy as np

from ..utilities import (
    is_distribution,
    normalize,
)


SparseArray = dict[tuple[int, ...], np.float128]


@overload
def greedy_mec(*marginals: np.ndarray, sparse: Literal[True]) -> SparseArray: ...


@overload
def greedy_mec(*marginals: np.ndarray, sparse: Literal[False]) -> np.ndarray: ...


# Type signature for default value of sparse
@overload
def greedy_mec(*marginals: np.ndarray) -> np.ndarray: ...


def greedy_mec(
    *marginals: np.ndarray, sparse: bool = False
) -> np.ndarray | SparseArray:
    """Greedily approximate minimum-entropy coupling for given marginals.

    Implements Algorithm 1 of https://arxiv.org/abs/1701.08254.

    Args:
        *marginals: 1D arrays representing marginals.
        sparse: Whether to return sparse representation.
            N.B. Do not change default value without appropriately updating
            return type of function signature for default value.

    Returns:
        Approximate minimum-entropy coupling represented as N-D array.
            The i-th dimension corresponds to the i-th marginal.
            If `sparse`, returns dictionary, otherwise numpy array.

    Raises:
        ValueError: If any input is not 1D numpy array.
        ValueError: If any input is not valid distribution.
        RuntimeError: If result is not valid coupling.

    Examples:
        >>> x = np.array([0.5, 0.5])
        >>> y = np.array([0.5, 0.5])
        >>> greedy_mec(x, y)
        array([[0.5, 0. ],
               [0. , 0.5]])

        >>> x = np.array([0.5, 0.5])
        >>> y = np.array([0.5, 0.5])
        >>> greedy_mec(x, y, sparse=True)
        {(0, 0): 0.5, (1, 1): 0.5}

        >>> x = np.array([1.0])
        >>> y = np.array([0.75, 0.25])
        >>> z = np.array([0.6, 0.3, 0.1])
        >>> greedy_mec(x, y, z)
        array([[[0.6 , 0.05, 0.1 ],
                [0.  , 0.25, 0.  ]]])

        >>> x = np.array([1.0])
        >>> y = np.array([0.75, 0.25])
        >>> z = np.array([0.6, 0.3, 0.1])
        >>> greedy_mec(x, y, z, sparse=True)
        {(0, 0, 0): 0.6, (0, 1, 1): 0.25, (0, 0, 2): 0.1, (0, 0, 1): 0.05}
    """
    # Validate marginals
    for i, marginal in enumerate(marginals):
        if not isinstance(marginal, np.ndarray):
            raise TypeError(f"Input {i} is not numpy array.")
        if not marginal.ndim == 1:
            raise ValueError(f"Input {i} is not 1D numpy array.")
        if not is_distribution(marginal):
            raise ValueError(f"Input {i} is not valid distribution.")

    # Validate sparse flag
    if not isinstance(sparse, bool):
        raise TypeError("Sparse flag must be boolean.")

    shapes = [len(marginal) for marginal in marginals]
    max_len = max(shapes)

    # Increase precision, renormalize, pad, and stack
    prepped_marginals = np.stack(
        [
            np.pad(
                normalize(marginal.astype(np.float128)), (0, max_len - len(marginal))
            )
            for marginal in marginals
        ]
    )

    # Set initial conditions
    coupling: np.ndarray | SparseArray
    if sparse:
        coupling = {}
    else:
        coupling = np.zeros(len(marginals) * [max_len], dtype=np.float128)

    # Main body of greedy approximation algorithm
    while True:
        # States where marginals place most mass
        idx = prepped_marginals.argmax(axis=1)
        # Maximum mass that can be taken from single state of all marginals
        mass = prepped_marginals[np.arange(prepped_marginals.shape[0]), idx].min()
        # Check stopping condition (marginals are exhausted)
        if mass == 0:
            break
        # Remove mass from marginals
        for i, j in enumerate(idx):
            prepped_marginals[i, j] -= mass
        # Add mass to coupling
        coupling[tuple(idx)] = mass

    # Remove padding if dense
    if isinstance(coupling, np.ndarray):
        return coupling[tuple(slice(0, bound) for bound in shapes)]
    return coupling

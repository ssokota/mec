from typing import TypeVar

import numpy as np


def entropy(distribution: np.ndarray) -> float:
    """Calculate entropy of given probability distribution.

    Args:
        distribution: Array representing probability distribution.

    Returns:
        Entropy of distribution.
    """
    mask = np.logical_not(np.isclose(distribution, 0))
    return -(distribution[mask] * np.log(distribution[mask])).sum()


def entropy_upper_bounds(z: np.ndarray, num_states: int) -> np.ndarray:
    """Compute upper bounds on entropies of distributions satisfying args.

    Args:
        z: Array of lower bounds on probability of one state.
        num_states: Number of states in distribution.

    Returns:
        Upper bounds on entropy of distributions satisfying args.
    """
    upper_bounds = np.ones_like(z) * np.log(num_states)
    # If lower bound greater than uniform, can beat trivial upper bound
    gt_uniform = z > 1 / num_states
    upper_bounds[gt_uniform] = -z[gt_uniform] * np.log(z[gt_uniform])
    remaining_mass = 1 - z
    remaining_mass_per_state = remaining_mass / (num_states - 1)
    # Mask out states with no remaining mass for numerical stability
    has_remaining_mass = ~np.isclose(remaining_mass, 0)
    mask = gt_uniform & has_remaining_mass
    upper_bounds[mask] -= remaining_mass[mask] * np.log(remaining_mass_per_state[mask])
    return upper_bounds


def get_proportional_rows(matrix: np.ndarray, row_index: int) -> np.ndarray:
    """Get rows in matrix proportional to given row.

    Args:
        matrix: Matrix to search.
        row_index: Index of row to compare to.

    Returns:
        Indices of rows in matrix that are proportional to row at row_index.
    """
    # Filter rows with inconsistent sparsity pattern and columns without entries.
    non_zero = matrix > 0
    row_indices = np.arange(matrix.shape[0])
    mask = (non_zero == non_zero[row_index]).all(axis=1)
    reduced_matrix = matrix[mask][:, non_zero[row_index]]
    reduced_row_indices = row_indices[mask]
    # Compute row-wise conditional probabilities
    normalized_matrix = reduced_matrix / reduced_matrix.sum(axis=1, keepdims=True)
    # Reduced rows whose conditional probabilities are equal to that of row_index
    isclose = np.isclose(
        normalized_matrix, normalized_matrix[reduced_row_indices == row_index]
    ).all(axis=1)
    # Return rows with close conditional probabilities
    return reduced_row_indices[np.where(isclose)[0]]


def greater_than_and_not_close(a: float, b: float) -> bool:
    """Check if `a` is greater than `b` and not approximately equal to `b`.

    Args:
        a: The first value to compare.
        b: The second value to compare.

    Returns:
        `a` is greater than `b` and not approximately equal to `b`.
    """
    return (a > b) and (not np.isclose(a, b))


def is_distribution(z: np.ndarray) -> bool:
    """Check if given array is valid probability distribution.

    Args:
        z: The array to check.

    Returns:
        True if `z` is a valid probability distribution, False otherwise.
    """
    return bool(
        np.logical_or(z >= 0, np.isclose(z, 0)).all() and np.isclose(z.sum(), 1)
    )


def is_deterministic(z: np.ndarray) -> bool:
    """Check if given array represents deterministic distribution.

    Args:
        z: The array to check.

    Returns:
        True if `z` is a deterministic distribution, False otherwise.
    """
    return np.isclose(z.sum(), 1) and np.sum(z == 1) == 1


def is_sublist(a: list[int], b: list[int]) -> bool:
    """Check if list `a` is sublist of list `b`.

    Args:
        a: List to check as sublist.
        b: List to check as superlist.

    Returns:
        True if `a` is sublist of `b`, False otherwise.
    """
    return all([a_ == b_ for a_, b_ in zip(a, b)])


TVector = TypeVar("TVector", float, np.float128, list, np.ndarray)


def log(t: TVector) -> TVector:
    """Compute natural logarithm of given value or array.

    Args:
        t: Value or array to compute natural logarithm of.

    Returns:
        Natural logarithm of `t`.
    """
    if isinstance(t, np.ndarray):
        mask = t > 0
        t[mask] = np.log(t[mask])
        t[~mask] = float("-inf")
        return t
    if isinstance(t, list):
        return [log(t_) for t_ in t]
    if t == 0:
        return type(t)("-inf")
    return np.log(t)


def normalize(propto: np.ndarray) -> np.ndarray:
    """Normalize given array to one.

    Args:
        propto: Array to normalize.

    Returns:
        Normalized array.
    """
    propto = propto.astype(np.float128)
    return propto / propto.sum()


def pad_to_length(arrays: list[np.ndarray], max_length: int) -> list[np.ndarray]:
    """Pad given arrays to given length with zeros.

    Args:
        arrays: Arrays to pad.
        max_length: Length to pad arrays to.

    Returns:
        Padded arrays.
    """
    return [
        np.concatenate([array, np.zeros(max_length - len(array))]) for array in arrays
    ]


def sample(z: np.ndarray) -> int:
    """Sample from given probability distribution.

    Args:
        z: Probability distribution to sample from.

    Returns:
        Index of sampled element.
    """
    # Note: np.random.choice does not support float128
    return np.random.choice(range(len(z)), p=z.astype(np.float64))

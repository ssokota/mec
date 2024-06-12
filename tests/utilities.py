import numpy as np


def generate_distribution(num_states: int, uniform: bool) -> np.ndarray:
    if uniform:
        return np.ones(num_states) / num_states
    x = np.random.normal(size=num_states)
    x = np.maximum(x, 0) + x.max() + 1e-6
    x = x / x.sum()
    return x

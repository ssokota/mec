"""In this example, we couple three marginals using greedy_mec."""

import numpy as np

from mec import greedy_mec
from mec.utilities import entropy

x = np.array([1.0])
y = np.array([0.75, 0.25])
z = np.array([0.6, 0.3, 0.1])
print(f"Marginal 1:\n{x}")
print(f"Marginal 2:\n{y}")
print(f"Marginal 3:\n{z}")
print(
    f"Lower bound on joint entropy of coupling: {max(entropy(x), entropy(y), entropy(z)):0.2f} nats"
)
print(
    f"Upper bound on joint entropy of coupling: {entropy(x) + entropy(y) + entropy(z):0.2f} nats"
)
coupling = greedy_mec(x, y, z)
assert np.isclose(coupling.sum(axis=(1, 2)), x).all()
assert np.isclose(coupling.sum(axis=(0, 2)), y).all()
assert np.isclose(coupling.sum(axis=(0, 1)), z).all()
print(f"Coupling:\n{coupling}")
print(f"Coupling shape:\n{coupling.shape}")
print(f"Joint entropy of coupling: {entropy(coupling):0.2f} nats")

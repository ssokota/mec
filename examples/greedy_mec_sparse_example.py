"""In this example, we use the sparse keyword to couple
marginals whose dense coupling representation would be too large to store.
"""

import numpy as np

from mec import greedy_mec
from mec.utilities import entropy, log


marginal_size = 10
num_marginals = 100
marginals = [np.ones(marginal_size) / marginal_size for _ in range(num_marginals)]
print("Marginal i:\n", marginals[0])
print("for i=1,...,100")
print(
    f"Lower bound on joint entropy of coupling: {max(entropy(m) for m in marginals):0.2f} nats"
)
print(
    f"Upper bound on joint entropy of coupling: {sum(entropy(m) for m in marginals):0.2f} nats"
)
coupling = greedy_mec(*marginals, sparse=True)
print(
    f"Number of entries dense representation would have: {marginal_size}^{num_marginals}"
)
print(f"Number of entries in sparse representation: {len(coupling)}")
print(
    f"Joint entropy of coupling: {-sum(p * log(p) for p in coupling.values()):0.2f} nats"
)

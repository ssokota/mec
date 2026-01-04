"""In this example, we use minimum entropy coupling to implement
entropic causal inference, an approach for determining the causal
direction between variables. Entropic causal inference is based
on an information-theoretic approach analogous to Occam's razor
that prefers the causal direction requiring less entropy.

Our example uses the following data-generating process:
1. X is detemined by an unobserved exogenous process.
2. Y is determined by X and an additional unobserved exogenous process.

One instance of such a process is a customer purchasing a product.
In this case, X is the customer and Y is the product purchase.
Since the customer is the cause of the product purchase, we would
like entropic causal inference to prefer X->Y.
"""

import numpy as np

from mec import greedy_mec
from mec.utilities import entropy


# Create the data-generating process.
dim = 100
# Marginal for X follows a distribution selected at uniform random from the simplex.
x_marginal = np.random.dirichlet(np.ones(dim))
# Conditional distribution Y|X=x is uniform among two randomly selected values of Y.
y_given_x = np.array([np.random.choice(dim, 2, replace=False) for _ in range(dim)])


# Generate samples from data-generating process.
num_samples = 10000
x_samples = np.random.choice(dim, num_samples, p=x_marginal)
y_samples = np.array([np.random.choice(y_given_x[x]) for x in x_samples])

# Calculate the empirical joint distribution.
joint_dist = np.zeros((dim, dim))
for x, y in zip(x_samples, y_samples):
    joint_dist[x, y] += 1 / num_samples
assert np.isclose(np.sum(joint_dist), 1)


def complexity(joint_dist: np.ndarray, row_as_cause: bool) -> float:
    """Compute the complexity of the causal direction.

    The complexity of X->Y is defined as: H(X) + H(MEC(Y|X_1, ..., Y|X_n)).
    The value H(MEC(Y|X_1, ..., Y|X_n)) is the minimum amount of entropy that
    an exogeneous variable E would need such Y could be a function of X and E.

    Args:
        joint_dist: The joint distribution of two variables.
        row_as_cause: Whether to compute complexity of row->col (else col->row).

    Returns:
        The complexity of the causal direction in nats.
    """
    joint_dist = joint_dist if row_as_cause else joint_dist.T
    marginal = np.sum(joint_dist, axis=1)
    non_zero = marginal > 0
    conditionals = joint_dist[non_zero] / marginal[non_zero, None]
    # Use sparse=True to accommodate large numbers of conditionals.
    coupling = greedy_mec(*conditionals, sparse=True)
    coupling_entropy = -sum(p * np.log(p) for p in coupling.values())
    return entropy(marginal) + coupling_entropy


# Since X mostly determines Y, the complexity of X->Y should be lower.
complexity_x_to_y = complexity(joint_dist, row_as_cause=True)
complexity_y_to_x = complexity(joint_dist, row_as_cause=False)
print(f"Complexity of X->Y: {complexity_x_to_y:2f} nats")
print(f"Complexity of Y->X: {complexity_y_to_x:2f} nats")
if complexity_x_to_y < complexity_y_to_x:
    print("Entropic causal inference prefers: X->Y.")
else:
    print("Entropic causal inference prefers: Y->X.")

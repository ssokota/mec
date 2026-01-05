from abc import ABC
from typing import Optional, TypeVar
import heapq

import numpy as np

from ..exceptions import ProbabilityZero
from ..utilities import entropy


class Posterior(ABC):
    """Abstract class for posterior distributions.

    Attributes:
        _distribution: Array representing probability distribution.
    """

    _distribution: np.ndarray

    def update_posterior(self, coupling_matrix: np.ndarray, observation: int) -> None:
        """Update posterior distribution based on observation.

        Args:
            coupling_matrix: Coupling matrix.
            observation: Observation to condition on.

        Raises:
            ProbabilityZero: If observation probability is zero.
        """
        unnormalized_conditional = coupling_matrix[observation]
        conditional_mass = coupling_matrix[observation].sum()
        if np.isclose(conditional_mass, 0):
            raise ProbabilityZero
        self.distribution = unnormalized_conditional / conditional_mass

    @property
    def distribution(self) -> np.ndarray:
        """Array representing probability distribution."""
        return self._distribution

    @distribution.setter
    def distribution(self, distribution: np.ndarray) -> None:
        """Set probability distribution."""
        # FactoredPosterior requires an in-place update here.
        np.copyto(self._distribution, distribution)


class TabularPosterior(Posterior):
    def __init__(self, prior: np.ndarray) -> None:
        """Posterior distribution represented as array.

        Args:
            prior: Prior distribution.

        Attributes:
            _distribution: Array representing probability distribution.
        """
        self._distribution = np.copy(prior)


class FactoredPosterior(Posterior):
    def __init__(self, prior: list[np.ndarray]) -> None:
        """Posterior distribution represented as list of component distributions.

        Joint probability is product of component distributions.

        Args:
            prior: List of prior distributions.

        Attributes:
            component_distributions: List of component distributions.
            _entropy_heap: Max heap of (-entropy, component_index) for efficient max queries.
        """
        self.component_distributions = [np.copy(x) for x in prior]
        self._entropy_heap = [
            (-entropy(dist), i) for i, dist in enumerate(self.component_distributions)
        ]
        heapq.heapify(self._entropy_heap)

    @property
    def max_entropy_component(self) -> int:
        """Index of component with maximum entropy (O(1) heap top access)."""
        return self._entropy_heap[0][1]

    @property
    def distribution(self) -> np.ndarray:
        """Array representing probability distribution of max entropy component."""
        return self.component_distributions[self.max_entropy_component]

    @distribution.setter
    def distribution(self, distribution: np.ndarray) -> None:
        """Set probability distribution of max entropy component and update heap."""
        max_ent_idx = self.max_entropy_component
        np.copyto(self.component_distributions[max_ent_idx], distribution)
        heapq.heapreplace(self._entropy_heap, (-entropy(distribution), max_ent_idx))


class AutoregressivePosterior(Posterior):
    PARENT_INDEX = -1

    def __init__(
        self,
        parent: Optional["AutoregressivePosterior"],
        prefix: list[int],
        prior: np.ndarray,
        depth: int,
        err_prob: float,
        is_leaf: bool,
    ) -> None:
        """Posterior distribution represented as node of prefix tree.

        Each node correponds to a prefix of the autoregressive distribution.
        Node holds probability of prefix and immediate suffixes.

        Args:
            parent: Parent posterior.
            prefix: Prefix for autoregressive posterior.
            prior: Prior distribution.
            depth: Depth of autoregressive posterior.
            err_prob: Probability prefix is wrong.
            is_leaf: Whether posterior is leaf.

        Attributes:
            parent: Parent posterior.
            prefix: Prefix for autoregressive posterior.
            _distribution: Probability distribution.
            alt_parent_index: Index of alternative parent as positive integer.
            depth: Depth of autoregressive posterior.
            is_leaf: Whether node is leaf.
            children: Dictionary of children posteriors.

        Constants:
            PARENT_INDEX: Index of parent in distribution.

        Notes:
            The parent is always the last element in the distribution.
            This is convenient because:
            1. The parent can be referred to by a constant index (-1).
            2. The child indices are consistent with the autoregressive distribution.
            However, this means the class must handle two indices for the parent, as it
            could be referred to either by its constant negative index or its actual index.
        """
        self.parent = parent
        self.prefix = prefix
        self._distribution = np.concatenate([np.copy(prior), np.array([0.0])])
        self.alt_parent_index = len(self.distribution) - 1
        self.enforce_constraint(self.PARENT_INDEX, err_prob)
        self.depth = depth
        self.is_leaf = is_leaf
        self.children: dict[int, AutoregressivePosterior] = {}

    def error_prob(self) -> float:
        """Return probability prefix is wrong.

        Returns:
            Probability prefix is wrong.
        """
        return self.distribution[self.PARENT_INDEX]

    def edge_expanded(self, edge: int) -> bool:
        """Check if edge is already expanded.

        Args:
            edge: Edge to check.

        Returns:
            True if edge is already expanded, False otherwise.
        """
        return (
            (edge == self.alt_parent_index)
            or (edge == self.PARENT_INDEX)
            or (edge in self.children)
        )

    def points_backward(self, edge: int) -> bool:
        """Check if edge points toward parent.

        Args:
            edge: Edge to check.

        Returns:
            True if edge is pointed toward parent, False otherwise.
        """
        return (edge == self.alt_parent_index) or (edge == self.PARENT_INDEX)

    def enforce_constraint(self, index_to_update: int, update_val: float) -> None:
        """Enforce constraint on probability distribution.

        Args:
            index_to_update: Index to update.
            update_val: Value to update index to.

        Raises:
            ProbabilityZero: If probability is zero.
        """
        # Check if posterior is delta distribution on probability zero event
        if np.isclose(update_val, 0) and self.distribution[index_to_update] == 1:
            raise ProbabilityZero
        # Otherwise, normalize distribution to remaining mass
        free_mass = self.distribution.sum() - self.distribution[index_to_update]
        if free_mass > 0:
            z = self.distribution * (1 - update_val) / free_mass
        else:
            z = np.zeros_like(self.distribution)
        z[index_to_update] = update_val
        self.distribution = z

    def clone(self) -> "AutoregressivePosterior":
        """Clone autoregressive posterior.

        Returns:
            Cloned autoregressive posterior.
        """
        return AutoregressivePosterior(
            self.parent,
            self.prefix,
            self.distribution[: self.PARENT_INDEX],
            self.depth,
            self.distribution[self.PARENT_INDEX],
            self.is_leaf,
        )

    def neighbor_prefix(self, edge: int) -> list[int]:
        """Return prefix of neighbor.

        Args:
            edge: Edge to neighbor.

        Returns:
            Prefix of neighbor.
        """
        if self.points_backward(edge):
            return self.prefix[: self.PARENT_INDEX]
        return self.prefix + [edge]


TPosterior = TypeVar(
    "TPosterior", TabularPosterior, FactoredPosterior, AutoregressivePosterior
)

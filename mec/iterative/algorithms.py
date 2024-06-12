from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional

import numpy as np

from ..exceptions import ProbabilityZero
from .marginals import (
    TMarginal,
    TabularMarginal,
    FactoredMarginal,
    AutoRegressiveMarginal,
    is_probability_zero,
)
from ..approximation.algorithms import greedy_mec
from .posteriors import (
    TPosterior,
    TabularPosterior,
    FactoredPosterior,
    AutoregressivePosterior,
)
from ..utilities import (
    entropy,
    entropy_upper_bound,
    get_proportional_rows,
    log,
    greater_than_and_not_close,
    is_deterministic,
    is_distribution,
    is_sublist,
    normalize,
    sample,
)

Helper = Callable[
    [TPosterior, list[int], np.ndarray], tuple[list[int], float, TPosterior]
]


def check_state(state: list[int]) -> None:
    """Check that state is list of integers.

    Args:
        state: State to check.

    Raises:
        TypeError: If state is not list of integers.
    """
    if not isinstance(state, list):
        raise TypeError("state is not list")
    if not all(isinstance(s, (int, np.integer)) for s in state):
        raise TypeError("state is not list of integers")


def check_positive_probability(state: list[int], marginal: TMarginal) -> None:
    """Check that state is supported by marginal.

    Args:
        state: State to check.
        marginal: Marginal distribution.

    Raises:
        ValueError: If state is not supported by marginal.
    """
    if is_probability_zero(marginal, state):
        raise ValueError("state is not supported by marginal")


class IMEC(ABC, Generic[TMarginal, TPosterior]):
    def __init__(
        self,
        mu: TMarginal,
        nu: AutoRegressiveMarginal,
        merge: bool = True,
    ):
        """Iteratively couple mu and nu such that joint distribution has low entropy.

        Implements Algorithm 3 of https://arxiv.org/pdf/2405.19540.

        Args:
            mu: Marginal distribution for X.
            nu: AutoRegressiveMarginal distribution for Y.
            merge: Whether to merge states inducing equivalent updates.
                Merging improves performance in some cases but increases runtime.

        Raises:
            TypeError: If nu or merge fail type checks.

        Attributes:
            mu: Marginal distribution.
            nu: AutoRegressiveMarginal distribution.
            merge: Whether to merge groups of states.
        """
        if not isinstance(nu, AutoRegressiveMarginal):
            raise TypeError("nu is not an AutoRegressiveMarginal")
        if not isinstance(merge, bool):
            raise TypeError("merge is not a boolean")
        self.mu: TMarginal = mu
        self.nu = nu
        self.merge = merge

    def sample(self) -> tuple[tuple[list[int], list[int]], float]:
        """Sample from joint distribution.

        Returns:
            Tuple of sampled states and log probability.
        """
        x, ll_x = self.mu.sample()
        y, ll_y_given_x = self.sample_y_given_x(x)
        return (x, y), ll_x + ll_y_given_x

    def evaluate(self, x: list[int], y: list[int]) -> float:
        """Evaluate log probability of joint state.

        Args:
            x: State for X.
            y: State for Y.

        Returns:
            Log probability of joint state.

        Raises:
            TypeError: If x or y is invalid type.
        """
        check_state(x)
        check_state(y)
        ll_x = self.mu.evaluate(x)
        ll_y_given_x = self.evaluate_y_given_x(y, x)
        return ll_x + ll_y_given_x

    def estimate_x_given_y(self, y: list[int]) -> tuple[list[int], float]:
        """Estimate X given Y.

        Args:
            y: State for Y.

        Returns:
            Tuple of estimated state and log probability.

        Raises:
            TypeError: If y is invalid type.
            ValueError: If y is not supported by nu.
        """
        check_state(y)
        check_positive_probability(y, self.nu)
        posterior = self._x_given_y(y)
        x, likelihoods = self._x_from_posterior(posterior, self._select_argmax_x_i)
        return x, sum(log(likelihoods))

    def sample_x_given_y(self, y: list[int]) -> tuple[list[int], float]:
        """Sample X given Y.

        Args:
            y: State for Y.

        Returns:
            Tuple of sampled state and log probability.

        Raises:
            TypeError: If y is invalid type
            ValueError: If y is not supported by nu.
        """
        check_state(y)
        check_positive_probability(y, self.nu)
        posterior = self._x_given_y(y)
        x, likelihoods = self._x_from_posterior(posterior, self._select_sample_x_i)
        return x, sum(log(likelihoods))

    def evaluate_x_given_y(self, x: list[int], y: list[int]) -> float:
        """Evaluate log probability of X given Y.

        Args:
            x: State for X.
            y: State for Y.

        Returns:
            Log probability of X given Y.

        Raises:
            TypeError: If x or y is invalid type.
            ValueError: If y is not supported by nu.
        """
        check_state(x)
        check_state(y)
        check_positive_probability(y, self.nu)
        posterior = self._x_given_y(y)
        try:
            x, likelihoods = self._x_from_posterior(
                posterior, self._make_select_true_x_i(x)
            )
            return sum(log(likelihoods))
        except ProbabilityZero:
            return float("-inf")

    def sample_y_given_x(self, x: list[int]) -> tuple[list[int], float]:
        """Sample Y given X.

        Args:
            x: State for X.

        Returns:
            Tuple of sampled state and log probability.

        Raises:
            TypeError: If x is invalid type.
            ValueError: If x is not supported by mu.
        """
        check_state(x)
        check_positive_probability(x, self.mu)
        helper = self._make_helper(self._select_sample_y_j, x=x, y=None)
        y, likelihoods, posterior = self._main_loop(helper)
        return y, sum(log(likelihoods))

    def estimate_y_given_x(self, x: list[int]) -> tuple[list[int], float]:
        """Estimate Y given X.

        Args:
            x: State for X.

        Returns:
            Tuple of estimated state and log probability.

        Raises:
            TypeError: If x is invalid type.
            ValueError: If x is not supported by mu.
        """
        check_state(x)
        check_positive_probability(x, self.mu)
        helper = self._make_helper(self._select_argmax_y_j, x=x, y=None)
        y, likelihoods, posterior = self._main_loop(helper)
        return y, sum(log(likelihoods))

    def evaluate_y_given_x(self, y: list[int], x: list[int]) -> float:
        """Evaluate log probability of Y given X.

        Args:
            y: State for Y.
            x: State for X.

        Returns:
            Log probability of Y given X.

        Raises:
            TypeError: If x or y is invalid type.
            ValueError: If x is not supported by mu.
        """
        check_state(x)
        check_state(y)
        check_positive_probability(x, self.mu)
        helper = self._make_helper(self._select_true_y_j, x=x, y=y)
        try:
            y, likelihoods, posterior = self._main_loop(helper)
            return sum(log(likelihoods))
        except ProbabilityZero:
            return float("-inf")

    def _x_given_y(self, y: list[int]) -> TPosterior:
        """Compute posterior distribution of X given Y.

        Args:
            y: State for Y.

        Returns:
            Posterior distribution of X given Y.
        """
        helper = self._make_helper(self._select_true_y_j, x=None, y=y)
        y, likelihoods, posterior = self._main_loop(helper)
        return posterior

    def _main_loop(
        self,
        helper: Helper,
    ) -> tuple[list[int], list[float], TPosterior]:
        """Main loop of IMEC.

        Args:
            helper: Helper function for iteration.

        Returns:
            Tuple of state, likelihoods, and posterior distribution.
        """
        posterior = self._initialize_posterior()
        y_prefix: list[int] = []
        likelihoods: list[float] = []
        while not self.nu.is_terminal(y_prefix):
            y_j_conditional = self._nu_conditional(y_prefix)
            y_j, y_j_likelihood, posterior = self._iterate(
                helper, posterior, y_prefix, y_j_conditional
            )
            y_prefix.append(y_j)
            likelihoods.append(y_j_likelihood)
        return y_prefix, likelihoods, posterior

    def _iterate(
        self,
        helper: Helper,
        posterior: TPosterior,
        y_prefix: list[int],
        y_j_conditional: np.ndarray,
    ) -> tuple[int, float, TPosterior]:
        """Body of IMEC iteration.

        Args:
            helper: Helper function for iteration.
            posterior: Posterior distribution for X.
            y_prefix: Prefix of Y.
            y_j_conditional: Conditional distribution of Y_j given prefix.

        Returns:
            Tuple of state, likelihood, and updated posterior.
        """
        y_j_likelihood = 1.0
        while True:
            # Compute maximum-entropy posterior partition
            posterior = self._max_ent_partition(posterior)
            # Couple partition with Y_j posterior, select Y_j group, update posterior
            y_j_group, group_likelihood, posterior = helper(
                posterior, y_prefix, y_j_conditional
            )
            # Update y_j likelihood
            y_j_likelihood *= group_likelihood
            # If group is singleton, iteration is over
            if len(y_j_group) == 1:
                break
            # If group not singleton, compute posterior over remaining Y_js
            y_j_conditional[~np.isin(np.arange(y_j_conditional.size), y_j_group)] = 0
            y_j_conditional /= y_j_conditional.sum()
        return y_j_group[0], y_j_likelihood, posterior

    def _make_helper(
        self,
        y_j_selector: Callable[[np.ndarray, list[int], Optional[list[int]]], int],
        x: Optional[list[int]] = None,
        y: Optional[list[int]] = None,
    ) -> Helper:
        """Create helper function for iteration.

        Args:
            y_j_selector: Function to select state of Y_j.
                If sampling Y, samples.
                If estimating Y, takes argmax.
                If evaluating Y, returns true value of Y_j.
            x: State for X.
            y: State for Y.

        Returns:
            Helper function for iteration.
        """

        def helper(
            posterior: TPosterior, y_prefix: list[int], y_j_conditional: np.ndarray
        ) -> tuple[list[int], float, TPosterior]:
            """Do coupling, select group of y_js, and update posterior.

            Args:
                posterior: Posterior distribution.
                y_prefix: Prefix of Y.
                y_j_conditional: Conditional distribution of Y_j given prefix.

            Returns:
                Tuple of state, likelihood, and updated posterior.

            Raises:
                ProbabilityZero: If posterior is zero.
            """
            # Handle deterministic posterior separately for stability
            if is_deterministic(posterior.distribution):
                y_j = y_j_selector(y_j_conditional, y_prefix, y)
                return [y_j], y_j_conditional[y_j], posterior

            # Couple, marginalize to get y_j posterior, select y_j, update X posterior.
            coupling_matrix = greedy_mec(y_j_conditional, posterior.distribution)
            y_j_posterior = self._marginalize(
                coupling_matrix, y_j_conditional, posterior, x
            )
            y_j = y_j_selector(y_j_posterior, y_prefix, y)
            if log(y_j_posterior[y_j]) == float("-inf"):
                raise ProbabilityZero
            posterior.update_posterior(coupling_matrix, y_j)

            # If not merging, return y_j as singleton group
            if not self.merge:
                return [y_j], y_j_posterior[y_j], posterior

            # If merging, compute all y_js that induce same posterior as selected y_j.
            y_j_group = get_proportional_rows(coupling_matrix, y_j)
            # Return all such y_js as group.
            return y_j_group.tolist(), y_j_posterior[y_j_group].sum(), posterior

        return helper

    def _nu_conditional(self, y: list[int]) -> np.ndarray:
        """Compute conditional distribution of Y_j given prefix.

        Args:
            y: Prefix of Y.

        Returns:
            Conditional distribution of Y_j given prefix.

        Raises:
            ValueError: If output of nu.conditional is not distribution.
        """
        conditional = self.nu.conditional(y)
        if not is_distribution(conditional):
            raise ValueError("Output of nu.conditional is not distribution")
        return normalize(conditional)

    @staticmethod
    def _select_argmax_y_j(
        y_j_posterior: np.ndarray,
        y_prefix: list[int],
        y: Optional[list[int]] = None,
    ) -> int:
        """Select Y_j by argmax.

        Args:
            y_j_posterior: Posterior distribution of Y_j.
            y_prefix: Prefix of Y.
            y: State for Y.

        Returns:
            Argmax over y_j_posterior.
        """
        return int(y_j_posterior.argmax())

    @staticmethod
    def _select_sample_y_j(
        y_j_posterior: np.ndarray,
        y_prefix: list[int],
        y: Optional[list[int]] = None,
    ) -> int:
        """Select Y_j by sampling.

        Args:
            y_j_posterior: Posterior distribution of Y_j.
            y_prefix: Prefix of Y.
            y: State for Y.

        Returns:
            Sample from y_j_posterior.
        """
        return sample(y_j_posterior)

    @staticmethod
    def _select_true_y_j(
        y_j_posterior: np.ndarray,
        y_prefix: list[int],
        y: Optional[list[int]] = None,
    ) -> int:
        """Select Y_j by true value.

        Args:
            y_j_posterior: Posterior distribution of Y_j.
            y_prefix: Prefix of Y.
            y: State for Y.

        Returns:
            True value of Y_j.
        """
        assert y is not None
        return y[len(y_prefix)]

    @staticmethod
    def _select_argmax_x_i(x_i_posterior: np.ndarray, x_prefix: list[int]) -> int:
        """Select X_i by argmax.

        Args:
            x_i_posterior: Posterior distribution of X_i.
            x_prefix: Prefix of X.

        Returns:
            Argmax over x_i_posterior.
        """
        return int(x_i_posterior.argmax())

    @staticmethod
    def _select_sample_x_i(x_i_posterior: np.ndarray, x_prefix: list[int]) -> int:
        """Select X_i by sampling.

        Args:
            x_i_posterior: Posterior distribution of X_i.
            x_prefix: Prefix of X.

        Returns:
            Sample from x_i_posterior.
        """
        return sample(x_i_posterior)

    @abstractmethod
    def _x_from_posterior(
        self, posterior: TPosterior, selector: Callable[[np.ndarray, list[int]], int]
    ) -> tuple[list[int], list[float]]:
        """Extract state and likelihood from posterior over X.

        Args:
            posterior: Posterior distribution over X.
            selector: Function to select state components.

        Returns:
            Tuple of state and likelihood.
        """

    @abstractmethod
    def _max_ent_partition(self, posterior: TPosterior) -> TPosterior:
        """Compute maximum-entropy posterior partition.

        Args:
            posterior: Posterior distribution over X.

        Returns:
            Maximum-entropy posterior partition.
        """

    @abstractmethod
    def _initialize_posterior(self) -> TPosterior:
        """Initialize posterior distribution.

        Returns:
            Initial posterior distribution.
        """

    @staticmethod
    @abstractmethod
    def _make_select_true_x_i(x: list[int]) -> Callable[[np.ndarray, list[int]], int]:
        """Create function to select index of X_i.

        Need abstract method for this because of ARIMEC's backtracking mechanism.

        Args:
            x: State for X.

        Returns:
            Function to select X_i.
        """

    @staticmethod
    @abstractmethod
    def _marginalize(
        coupling_matrix: np.ndarray,
        y_j_conditional: np.ndarray,
        posterior: TPosterior,
        x: Optional[list[int]] = None,
    ) -> np.ndarray:
        """Marginalize coupling matrix to get posterior over Y_j.

        Need abstract method for this because of ARIMEC's backtracking mechanism.
        y_j_conditional is included as argument for computational efficiency.

        Args:
            coupling_matrix: Coupling matrix.
            y_j_conditional: Conditional distribution of Y_j.
            posterior: Posterior distribution over X.
            x: State for X.

        Returns:
            Posterior distribution over Y_j.
        """


class TIMEC(IMEC[TabularMarginal, TabularPosterior]):
    def __init__(
        self,
        mu: TMarginal,
        nu: AutoRegressiveMarginal,
        merge: bool = True,
    ):
        """Overrides base class to check mu.

        Implements Algorithm 1 of https://arxiv.org/pdf/2405.19540.

        Raises:
            TypeError: If mu is not TabularMarginal.
            ValueError: If mu.distribution is not valid distribution.
        """
        if not isinstance(mu, TabularMarginal):
            raise TypeError("mu is not TabularMarginal")
        if not is_distribution(mu.distribution):
            raise ValueError("mu.distribution is not valid distribution")
        super().__init__(mu, nu, merge)

    def _x_from_posterior(
        self,
        posterior: TabularPosterior,
        selector: Callable[[np.ndarray, list[int]], int],
    ) -> tuple[list[int], list[float]]:
        """Implements method from base class.

        Directly applies selector to posterior distribution.
        """
        x = selector(posterior.distribution, [])
        likelihood = posterior.distribution[x]
        return [x], [likelihood]

    def _max_ent_partition(self, posterior: TabularPosterior) -> TabularPosterior:
        """Implements method from base class.

        Directly returns posterior.
        """
        return posterior

    def _initialize_posterior(self) -> TabularPosterior:
        """Implements method from base class."""
        return TabularPosterior(self.mu.distribution)

    @staticmethod
    def _make_select_true_x_i(x: list[int]) -> Callable[[np.ndarray, list[int]], int]:
        """Implements method from base class."""

        def select_true_x_i(distribution: np.ndarray, x_prefix: list[int]) -> int:
            """Directly return singleton value of x.

            Args:
                distribution: Posterior distribution.
                x_prefix: Prefix of X.

            Returns:
                Singleton element of x.
            """
            return x[0]

        return select_true_x_i

    @staticmethod
    def _marginalize(
        coupling_matrix: np.ndarray,
        y_j_conditional: np.ndarray,
        posterior: TabularPosterior,
        x: Optional[list[int]] = None,
    ) -> np.ndarray:
        """Implements method from base class."""
        if x is None:
            return y_j_conditional
        return normalize(coupling_matrix[:, x[0]])


class FIMEC(IMEC[FactoredMarginal, FactoredPosterior]):
    def __init__(
        self,
        mu: TMarginal,
        nu: AutoRegressiveMarginal,
        merge: bool = True,
    ):
        """Overrides base class to check mu.

        Implements Algorithm 2 of https://arxiv.org/pdf/2405.19540.

        Raises:
            TypeError: If mu is not FactoredMarginal.
            ValueError: If mu.component_distributions has invalid distributions.
        """
        if not isinstance(mu, FactoredMarginal):
            raise TypeError("mu is not a FactoredMarginal")
        if not all(
            is_distribution(component) for component in mu.component_distributions
        ):
            raise ValueError("mu.component_distributions has invalid distributions")
        super().__init__(mu, nu, merge)

    def _x_from_posterior(
        self,
        posterior: FactoredPosterior,
        selector: Callable[[np.ndarray, list[int]], int],
    ) -> tuple[list[int], list[float]]:
        """Implements method from base class.

        Iterates through components of posterior to select states.
        """
        x_prefix: list[int] = []
        likelihoods: list[float] = []
        for component_distribution in posterior.component_distributions:
            x_i = selector(component_distribution, x_prefix)
            x_prefix.append(x_i)
            likelihoods.append(component_distribution[x_i])
        return x_prefix, likelihoods

    def _max_ent_partition(self, posterior: FactoredPosterior) -> FactoredPosterior:
        """Implements method from base class.

        Selects component with highest entropy.
        """
        entropies = [
            entropy(component_distribution)
            for component_distribution in posterior.component_distributions
        ]
        highest_entropy_component = int(np.argmax(entropies))
        return FactoredPosterior(
            posterior.component_distributions, highest_entropy_component
        )

    def _initialize_posterior(self) -> FactoredPosterior:
        """Implements method from base class."""
        return FactoredPosterior(self.mu.component_distributions, 0)

    @staticmethod
    def _make_select_true_x_i(x: list[int]) -> Callable[[np.ndarray, list[int]], int]:
        """Implements method from base class."""

        def select_true_x_i(distribution: np.ndarray, x_prefix: list[int]) -> int:
            """Return subsequent component of x given x_prefix.

            Args:
                distribution: Posterior distribution.
                x_prefix: Prefix of X.

            Returns:
                Subsequent component of x.
            """
            return x[len(x_prefix)]

        return select_true_x_i

    @staticmethod
    def _marginalize(
        coupling_matrix: np.ndarray,
        y_j_conditional: np.ndarray,
        posterior: FactoredPosterior,
        x: Optional[list[int]] = None,
    ) -> np.ndarray:
        """Implements method from base class."""
        if x is None:
            return y_j_conditional
        return normalize(coupling_matrix[:, x[posterior.active_component]])


class ARIMEC(IMEC[AutoRegressiveMarginal, AutoregressivePosterior]):
    def __init__(
        self,
        mu: TMarginal,
        nu: AutoRegressiveMarginal,
        merge: bool = True,
    ):
        """Overrides base class to type check mu.

        Implements Definition 4.5 of https://arxiv.org/pdf/2405.19540.

        Raises:
            TypeError: If mu is not AutoRegressiveMarginal.
        """
        if not isinstance(mu, AutoRegressiveMarginal):
            raise TypeError("mu is not a AutoRegressiveMarginal")
        super().__init__(mu, nu, merge)

    def _x_from_posterior(
        self,
        posterior: AutoregressivePosterior,
        x_i_selector: Callable[[np.ndarray, list[int]], int],
    ) -> tuple[list[int], list[float]]:
        """Implements method from base class.

        Traverses tree to select state for x. Traversal occurs in two stages:
        1. Take path through tree until leaf node is reached.
        2. Sample X from posterior given leaf node.
        """
        likelihoods: list[float] = []
        while True:
            prefix = posterior.prefix
            # Already visited leaf node reached, continue to stage two.
            if posterior.is_leaf:
                x = prefix
                break
            # Select edge to traverse, update likelihood.
            edge = x_i_selector(posterior.distribution, posterior.prefix)
            likelihoods.append(posterior.distribution[edge])
            # Not-yet visited leaf node reached, continue to stage two.
            if not posterior.edge_expanded(edge):
                x = prefix + [edge]
                break
            # Step to next node, update posterior appropriately.
            if posterior.points_backward(edge):
                assert posterior.parent is not None
                posterior = posterior.parent
                posterior.enforce_constraint(
                    prefix[AutoregressivePosterior.PARENT_INDEX], 0
                )
            else:
                posterior = posterior.children[edge]
                posterior.enforce_constraint(AutoregressivePosterior.PARENT_INDEX, 0)
        # Sample X from posterior given leaf node.
        while not self.mu.is_terminal(x):
            conditional = self._mu_conditional(x)
            edge = x_i_selector(conditional, x)
            x += [edge]
            likelihoods.append(conditional[edge])
        return x, likelihoods

    def _max_ent_partition(
        self, posterior: AutoregressivePosterior
    ) -> AutoregressivePosterior:
        """Implements method from base class.

        Traverses tree to find node with highest entropy. Traversal works by
        performing depth-first search with pruning. It is depth first in the sense of
        the tree rooted at the input posterior. Pruning uses an upper bound on
        the entropy of all nodes reachable from a particular edge.
        """
        max_ent = entropy(posterior.distribution)
        max_ent_posterior = posterior
        queue = [posterior]
        searched = set([])
        while len(queue) > 0:
            posterior = queue.pop()
            # If leaf node or already visited, continue.
            if posterior.is_leaf or tuple(posterior.prefix) in searched:
                continue
            # If entropy is greater than current max, update max.
            searched.add(tuple(posterior.prefix))
            ent = entropy(posterior.distribution)
            if greater_than_and_not_close(ent, max_ent):
                max_ent_posterior = posterior
                max_ent = ent
            # If entropy upper bound is less than current max entropy,
            # prune subtree or subtree complement.
            for edge, edge_val in enumerate(posterior.distribution):
                if greater_than_and_not_close(
                    entropy_upper_bound(1 - edge_val, self.mu.branching_factor + 1),
                    max_ent,
                ):
                    queue.append(self._tree_step(posterior, edge, edge_val))
        return max_ent_posterior

    def _tree_step(
        self, posterior: AutoregressivePosterior, edge: int, edge_val: float
    ) -> AutoregressivePosterior:
        """Step from node to prefix tree to neighboring node.

        Args:
            posterior: Current node.
            edge: Edge to traverse.
            edge_val: Probability value of edge.

        Returns:
            Posterior assocaited with neighboring node in tree.
        """
        # Check if we've traversed this edge already.
        if posterior.edge_expanded(edge):
            # Handle parent and child cases separately.
            if posterior.points_backward(edge):
                parent = posterior.parent
                assert parent is not None
                edge_from_parent_to_current = posterior.prefix[-1]
                parent.enforce_constraint(edge_from_parent_to_current, 1 - edge_val)
                return parent
            child = posterior.children[edge]
            edge_from_current_to_child = AutoregressivePosterior.PARENT_INDEX
            child.enforce_constraint(edge_from_current_to_child, 1 - edge_val)
            return child
        # If not, construct prefix for neighboring node.
        # Note that the edge being untouched guarantees it's pointing to child.
        new_x_prefix = posterior.prefix + [edge]
        is_leaf = self.mu.is_terminal(new_x_prefix)
        # If leaf node, make posterior, otherwise ask mu.
        if is_leaf:
            distribution = np.array([1])
        else:
            distribution = self._mu_conditional(new_x_prefix)
        # Update origin node and return neighboring node.
        posterior.children[edge] = AutoregressivePosterior(
            posterior,
            new_x_prefix,
            distribution,
            posterior.depth + 1,
            1 - edge_val,
            is_leaf,
        )
        return posterior.children[edge]

    def _initialize_posterior(self) -> AutoregressivePosterior:
        """Implements method from base class."""
        return AutoregressivePosterior(
            parent=None,
            prefix=[],
            prior=self._mu_conditional([]),
            depth=0,
            err_prob=0,
            is_leaf=False,
        )

    @staticmethod
    def _make_select_true_x_i(x: list[int]) -> Callable[[np.ndarray, list[int]], int]:
        """Implements method from base class."""

        def select_true_x_i(posterior: np.ndarray, x_prefix: list[int]) -> int:
            """Select true edge of prefix tree.

            Args:
                posterior: Posterior distribution.
                x_prefix: Prefix of X.

            Returns:
                Parent index if x_prefix wrong, otherwise subsequent component of x.
            """
            if is_sublist(x_prefix, x):
                return x[len(x_prefix)]
            return AutoregressivePosterior.PARENT_INDEX

        return select_true_x_i

    @staticmethod
    def _marginalize(
        coupling_matrix: np.ndarray,
        y_j_conditional: np.ndarray,
        posterior: AutoregressivePosterior,
        x: Optional[list[int]] = None,
    ) -> np.ndarray:
        """Implements method from base class."""
        if x is None:
            return y_j_conditional
        x_i = (
            x[posterior.depth]
            if is_sublist(posterior.prefix, x)
            else posterior.PARENT_INDEX
        )
        return normalize(coupling_matrix[:, x_i])

    def _mu_conditional(self, x: list[int]) -> np.ndarray:
        """Compute conditional distribution of X_i given prefix.

        Args:
            x: Prefix of X.

        Returns:
            Conditional distribution of X_i given prefix.

        Raises:
            ValueError: If output of mu.conditional is not distribution.
        """
        conditional = self.mu.conditional(x)
        if not is_distribution(conditional):
            raise ValueError("Output of mu.conditional is not distribution")
        return normalize(conditional)

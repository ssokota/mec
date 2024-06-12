from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np


class SupportsEvaluate(Protocol):
    """Supports evaluation of log probability of a state"""

    def evaluate(self, z: list[int]) -> float:
        """Evaluate log probability of `z`.

        Args:
            z: State to evaluate.

        Returns:
            Log probability of `z`.
        """


class SupportsSample(Protocol):
    """Supports sampling from the distribution"""

    def sample(self) -> tuple[list[int], float]:
        """Sample from the distribution.

        Returns:
            Tuple of sampled state and its log probability.
        """


def is_probability_zero(evaluate_supporter: SupportsEvaluate, z: list[int]) -> bool:
    """Check if probability of `z` is zero

    Args:
        evaluate_supporter: Object supporting evaluation.
        z: State to evaluate.

    Returns:
        True if probability of `z` is zero, False otherwise.
    """
    return evaluate_supporter.evaluate(z) == float("-inf")


@runtime_checkable
class TabularMarginal(SupportsSample, SupportsEvaluate, Protocol):
    """Distribution represented as a table of probabilities.

    Attributes:
        distribution: Array of probabilities.
    """

    distribution: np.ndarray


@runtime_checkable
class FactoredMarginal(SupportsSample, SupportsEvaluate, Protocol):
    """Distribution represented as a product of component distributions.

    Attributes:
        component_distributions: List of component distributions.
    """

    component_distributions: list[np.ndarray]


@runtime_checkable
class AutoRegressiveMarginal(SupportsSample, SupportsEvaluate, Protocol):
    """Distribution represented as autoregressive conditional distributions.

    Attributes:
        branching_factor: Maximum number of outcomes of conditional distributions.
    """

    branching_factor: int

    def conditional(self, prefix: list[int]) -> np.ndarray:
        """Conditional distribution for `prefix`

        Args:
            prefix: Prefix to condition on.

        Returns:
            Conditional distribution.
        """

    def is_terminal(self, prefix: list[int]) -> bool:
        """Whether `prefix` is complete"""


TTabularOrFactoredMarginal = TypeVar(
    "TTabularOrFactoredMarginal", TabularMarginal, FactoredMarginal
)
TMarginal = TypeVar(
    "TMarginal", TabularMarginal, FactoredMarginal, AutoRegressiveMarginal
)


class ConvertedMarginal(ABC, Generic[TTabularOrFactoredMarginal]):
    """Wrapper for converting between different marginal types."""

    def sample(self) -> tuple[list[int], float]:
        """Implements method from `SupportsSample` protocol."""
        return self.marginal.sample()

    def evaluate(self, z: list[int]) -> float:
        """Implements method from `SupportsEvaluate` protocol."""
        return self.marginal.evaluate(z)

    @property
    @abstractmethod
    def marginal(self) -> TTabularOrFactoredMarginal:
        """Underlying marginal distribution."""


class FactoredFromTabular(ConvertedMarginal[TabularMarginal]):
    def __init__(self, marginal: TabularMarginal):
        """Convert tabular marginal to factored marginal.

        Args:
            marginal: Tabular marginal to convert.

        Attributes:
            _marginal: Tabular marginal.
            component_distributions: List of component distributions.
        """
        self._marginal = marginal
        self.component_distributions = [marginal.distribution]

    @property
    def marginal(self) -> TabularMarginal:
        """Implements method from `ConvertedMarginal` ABC."""
        return self._marginal


class AutoRegressiveFromTabular(ConvertedMarginal[TabularMarginal]):
    def __init__(self, marginal: TabularMarginal):
        """Convert tabular marginal to autoregressive marginal.

        Args:
            marginal: Tabular marginal to convert.

        Attributes:
            _marginal: Tabular marginal.
            branching_factor: Maximum number of outcomes of conditional distributions.
        """
        self._marginal = marginal
        self.branching_factor = len(marginal.distribution)

    def conditional(self, prefix: list[int]) -> np.ndarray:
        """Implements method from `AutoRegressiveMarginal` protocol.

        Raises:
            ValueError: If `prefix` is non-trivial
        """
        if len(prefix) > 0:
            raise ValueError("Non-trivial prefix")
        return self.marginal.distribution

    def is_terminal(self, prefix: list[int]) -> bool:
        """Implement method from `AutoRegressiveMarginal` protocol.

        Raises:
            ValueError: If `prefix` is non-trivial
        """
        if len(prefix) > 1:
            raise ValueError("Prefix too long")
        return len(prefix) == 1

    @property
    def marginal(self) -> TabularMarginal:
        """Implements method from `ConvertedMarginal` ABC."""
        return self._marginal


class AutoRegressiveFromFactored(ConvertedMarginal[FactoredMarginal]):
    def __init__(self, marginal: FactoredMarginal):
        """Convert factored marginal to autoregressive marginal.

        Args:
            marginal: Factored marginal to convert.

        Attributes:
            _marginal: Factored marginal.
            branching_factor: Maximum number of outcomes of conditional distributions.
        """
        self._marginal = marginal
        self.branching_factor = max(
            [
                len(component_distribution)
                for component_distribution in marginal.component_distributions
            ]
        )

    def conditional(self, prefix: list[int]) -> np.ndarray:
        """Implements method from `AutoRegressiveMarginal` protocol.

        Raises:
            ValueError: If `prefix` too long
        """
        if len(prefix) > len(self.marginal.component_distributions) - 1:
            raise ValueError("Prefix too long")
        return self.marginal.component_distributions[len(prefix)]

    def is_terminal(self, prefix: list[int]) -> bool:
        """Implement method from `AutoRegressiveMarginal` protocol.

        Raises:
            ValueError: If `prefix` too long
        """
        if len(prefix) > len(self.marginal.component_distributions):
            raise ValueError("Prefix too long")
        return len(prefix) == len(self.marginal.component_distributions)

    @property
    def marginal(self) -> FactoredMarginal:
        """Implements method from `ConvertedMarginal` ABC."""
        return self._marginal

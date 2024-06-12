import itertools
from typing import TypeVar

import numpy as np

from mec.utilities import log, sample

from utilities import generate_distribution


class DummyTabularMarginal:
    def __init__(self, num_states: int, uniform: bool):
        self.distribution = generate_distribution(num_states, uniform)

    def evaluate(self, x: list[int]) -> float:
        return log(self.distribution[x[0]])

    def sample(self) -> tuple[list[int], float]:
        x_i = sample(self.distribution)
        return [x_i], log(self.distribution[x_i])

    def enumerate(self) -> list[list[int]]:
        return [[x_i] for x_i in range(len(self.distribution))]


class DummyFactoredMarginal:
    def __init__(
        self, num_states_per_component: int, num_components: int, uniform: bool
    ):
        self.component_distributions = [
            generate_distribution(num_states_per_component, uniform)
            for _ in range(num_components)
        ]

    def evaluate(self, x: list[int]) -> float:
        return sum(
            log([self.component_distributions[i][x_i] for i, x_i in enumerate(x)])
        )

    def sample(self) -> tuple[list[int], float]:
        log_likelihood = 0
        x = []
        for distribution in self.component_distributions:
            x_i = sample(distribution)
            log_likelihood += log(distribution[x_i])
            x.append(x_i)
        return x, log_likelihood

    def enumerate(self) -> list[list[int]]:
        return [
            list(x)
            for x in itertools.product(
                list(range(len(self.component_distributions[0]))),
                repeat=len(self.component_distributions),
            )
        ]


class DummyAutoRegressiveMarginal:
    def __init__(self, num_states_per_node: int, depth: int, uniform: bool):
        self.num_states_per_node = num_states_per_node
        self.branching_factor = num_states_per_node
        self.depth = depth
        nodes = []
        for d in range(depth + 1):
            nodes += [
                list(x)
                for x in itertools.product(
                    list(range(self.num_states_per_node)),
                    repeat=d,
                )
            ]
        self.distributions = {
            tuple(node): generate_distribution(num_states_per_node, uniform)
            for node in nodes
        }

    def sample(self) -> tuple[list[int], float]:
        z: list[int] = []
        log_likelihood = 0
        for _ in range(self.depth):
            z_i = sample(self.distributions[tuple(z)])
            log_likelihood += log(self.distributions[tuple(z)][z_i])
            z.append(z_i)
        return z, log_likelihood

    def conditional(self, prefix: list[int]) -> np.ndarray:
        return self.distributions[tuple(prefix)]

    def is_terminal(self, prefix: list[int]) -> bool:
        assert len(prefix) <= self.depth
        return len(prefix) == self.depth

    def evaluate(self, z: list[int]) -> float:
        ll = 0
        for upper in range(len(z)):
            prefix = tuple(z[:upper])
            ll += log(self.distributions[prefix][z[upper]])
        return ll

    def enumerate(self) -> list[list[int]]:
        return [
            list(x)
            for x in itertools.product(
                list(range(self.num_states_per_node)),
                repeat=self.depth,
            )
        ]


TDummy = TypeVar(
    "TDummy", DummyTabularMarginal, DummyFactoredMarginal, DummyAutoRegressiveMarginal
)
TDummy_F_or_AR = TypeVar(
    "TDummy_F_or_AR", DummyFactoredMarginal, DummyAutoRegressiveMarginal
)

DummyMarginal = (
    DummyTabularMarginal | DummyFactoredMarginal | DummyAutoRegressiveMarginal
)

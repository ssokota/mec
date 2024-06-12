import unittest

import numpy as np

from mec import FIMEC

from marginals import DummyAutoRegressiveMarginal, DummyFactoredMarginal


class MergeTest(unittest.TestCase):
    def test_merge_helps(self):
        for uniform in [True, False]:
            x_marginal = DummyFactoredMarginal(
                num_states_per_component=2, num_components=3, uniform=uniform
            )
            y_ar_marginal = DummyAutoRegressiveMarginal(
                num_states_per_node=4, depth=2, uniform=uniform
            )
            ent = {True: 0, False: 0}
            for merge in [True, False]:
                imec = FIMEC(x_marginal, y_ar_marginal, merge=merge)
                for x in x_marginal.enumerate():
                    for y in y_ar_marginal.enumerate():
                        log_likelihood = imec.evaluate(x, y)
                        if log_likelihood > float("-inf"):
                            ent[merge] -= np.exp(log_likelihood) * log_likelihood
            self.assertLess(ent[True], ent[False])


if __name__ == "__main__":
    unittest.main()

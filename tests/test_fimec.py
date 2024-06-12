import unittest

from mec import FIMEC

from common import (
    run_methods,
    assert_marginalizes,
    assert_consistency,
    assert_is_distribution,
    assert_generalization,
)
from marginals import (
    DummyTabularMarginal,
    DummyFactoredMarginal,
    DummyAutoRegressiveMarginal,
)
from utilities import generate_distribution


class FIMECTest(unittest.TestCase):
    def test_fimec_runs(self):
        for uniform in [True, False]:
            for merge in [True, False]:
                x_f_marginal = DummyFactoredMarginal(
                    num_states_per_component=2, num_components=2, uniform=uniform
                )
                y_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=2, uniform=uniform
                )
                imec = FIMEC(x_f_marginal, y_ar_marginal, merge=merge)
                run_methods(imec)

    def test_fimec_is_distribuion(self):
        for uniform in [True, False]:
            for merge in [True, False]:
                x_f_marginal = DummyFactoredMarginal(
                    num_states_per_component=2, num_components=2, uniform=uniform
                )
                y_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=2, uniform=uniform
                )
                imec = FIMEC(x_f_marginal, y_ar_marginal, merge=merge)
                assert_is_distribution(self, imec)

    def test_fimec_marginalizes(self):
        for uniform in [True, False]:
            for merge in [True, False]:
                x_f_marginal = DummyFactoredMarginal(
                    num_states_per_component=2, num_components=2, uniform=uniform
                )
                y_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=2, uniform=uniform
                )
                imec = FIMEC(x_f_marginal, y_ar_marginal, merge=merge)
                assert_marginalizes(self, imec, x_f_marginal, y_ar_marginal)

    def test_fimec_consistent(self):
        for uniform in [True, False]:
            for merge in [True, False]:
                x_f_marginal = DummyFactoredMarginal(
                    num_states_per_component=2, num_components=2, uniform=uniform
                )
                y_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=2, uniform=uniform
                )
                imec = FIMEC(x_f_marginal, y_ar_marginal, merge=merge)
                assert_consistency(self, imec, x_f_marginal, y_ar_marginal)

    def test_fimec_generalizes_timec(self):
        num_states_per_component = 5
        num_components = 1
        num_states_y = 4
        depth = 2
        for uniform in [False, True]:
            y_ar_marginal = DummyAutoRegressiveMarginal(
                num_states_per_node=num_states_y, depth=depth, uniform=uniform
            )
            distribution = generate_distribution(num_states_per_component, uniform)
            x_marginal = DummyTabularMarginal(num_states_per_component, uniform)
            x_f_marginal = DummyFactoredMarginal(
                num_states_per_component, num_components, uniform
            )
            x_marginal.distribution = distribution
            x_f_marginal.component_distributions = [distribution]
            assert_generalization(self, y_ar_marginal, x_marginal, x_f_marginal)

    def test_fimec_raises(self):
        x_f_marginal = DummyFactoredMarginal(
            num_states_per_component=2, num_components=2, uniform=True
        )
        y_ar_marginal = DummyAutoRegressiveMarginal(
            num_states_per_node=4, depth=2, uniform=True
        )
        with self.assertRaises(TypeError):
            FIMEC(y_ar_marginal, y_ar_marginal)
        with self.assertRaises(TypeError):
            FIMEC(x_f_marginal, x_f_marginal)
        with self.assertRaises(TypeError):
            FIMEC(x_f_marginal, y_ar_marginal, merge="invalid")
        imec = FIMEC(x_f_marginal, y_ar_marginal)
        with self.assertRaises(TypeError):
            imec.evaluate([0], "invalid")
        with self.assertRaises(TypeError):
            imec.evaluate("invalid", [0])
        with self.assertRaises(TypeError):
            imec.evaluate_x_given_y([0], "invalid")
        with self.assertRaises(TypeError):
            imec.evaluate_x_given_y("invalid", [0])
        with self.assertRaises(TypeError):
            imec.evaluate_y_given_x([0], "invalid")
        with self.assertRaises(TypeError):
            imec.evaluate_y_given_x("invalid", [0])
        with self.assertRaises(TypeError):
            imec.estimate_x_given_y("invalid")
        with self.assertRaises(TypeError):
            imec.estimate_y_given_x("invalid")
        with self.assertRaises(TypeError):
            imec.sample_x_given_y("invalid")
        with self.assertRaises(TypeError):
            imec.sample_y_given_x("invalid")


if __name__ == "__main__":
    unittest.main()

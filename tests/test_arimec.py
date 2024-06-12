import unittest

from mec import ARIMEC

from common import (
    run_methods,
    assert_marginalizes,
    assert_consistency,
    assert_is_distribution,
    assert_generalization,
)
from marginals import (
    DummyTabularMarginal,
    DummyAutoRegressiveMarginal,
)
from utilities import generate_distribution


class ARIMECTest(unittest.TestCase):
    def test_arimec_runs(self):
        for uniform in [True, False]:
            for merge in [True, False]:
                x_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=2, uniform=uniform
                )
                y_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=3, uniform=uniform
                )
                imec = ARIMEC(x_ar_marginal, y_ar_marginal, merge=merge)
                run_methods(imec)

    def test_arimec_is_distribuion(self):
        for uniform in [True, False]:
            for merge in [True, False]:
                x_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=2, uniform=uniform
                )
                y_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=3, uniform=uniform
                )
                imec = ARIMEC(x_ar_marginal, y_ar_marginal, merge=merge)
                assert_is_distribution(self, imec)

    def test_arimec_marginalizes(self):
        for uniform in [True, False]:
            for merge in [True, False]:
                x_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=2, uniform=uniform
                )
                y_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=3, uniform=uniform
                )
                imec = ARIMEC(x_ar_marginal, y_ar_marginal, merge=merge)
                assert_marginalizes(self, imec, x_ar_marginal, y_ar_marginal)

    def test_arimec_consistent(self):
        for uniform in [True, False]:
            for merge in [True, False]:
                x_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=2, uniform=uniform
                )
                y_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=3, uniform=uniform
                )
                imec = ARIMEC(x_ar_marginal, y_ar_marginal, merge=merge)
                assert_consistency(self, imec, x_ar_marginal, y_ar_marginal)

    def test_arimec_generalizes_timec(self):
        num_states_x = 5
        depth_x = 1
        num_states_y = 4
        depth_y = 2
        for uniform in [False, True]:
            y_ar_marginal = DummyAutoRegressiveMarginal(
                num_states_per_node=num_states_y, depth=depth_y, uniform=uniform
            )
            distribution = generate_distribution(num_states_x, uniform)
            x_marginal = DummyTabularMarginal(num_states_x, uniform)
            x_r_marginal = DummyAutoRegressiveMarginal(num_states_x, depth_x, uniform)
            x_marginal.distribution = distribution
            x_r_marginal.distributions = {(): distribution}
            assert_generalization(self, y_ar_marginal, x_marginal, x_r_marginal)

    def test_arimec_raises(self):
        x_marginal = DummyTabularMarginal(4, True)
        x_ar_marginal = DummyAutoRegressiveMarginal(
            num_states_per_node=4, depth=2, uniform=True
        )
        y_ar_marginal = DummyAutoRegressiveMarginal(
            num_states_per_node=4, depth=2, uniform=True
        )
        with self.assertRaises(TypeError):
            ARIMEC(x_marginal, y_ar_marginal)
        with self.assertRaises(TypeError):
            ARIMEC(x_ar_marginal, x_marginal)
        with self.assertRaises(TypeError):
            ARIMEC(x_ar_marginal, y_ar_marginal, merge="invalid")
        imec = ARIMEC(x_ar_marginal, y_ar_marginal)
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

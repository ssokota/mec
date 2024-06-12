import unittest

from mec import TIMEC

from common import (
    run_methods,
    assert_marginalizes,
    assert_consistency,
    assert_is_distribution,
)
from marginals import DummyTabularMarginal, DummyAutoRegressiveMarginal


class TIMECTest(unittest.TestCase):
    def test_timec_runs(self):
        for uniform in [True, False]:
            for merge in [True, False]:
                x_marginal = DummyTabularMarginal(num_states=5, uniform=uniform)
                y_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=2, uniform=uniform
                )
                imec = TIMEC(x_marginal, y_ar_marginal, merge=merge)
                run_methods(imec)

    def test_timec_is_distribuion(self):
        for uniform in [True, False]:
            for merge in [True, False]:
                x_marginal = DummyTabularMarginal(num_states=5, uniform=uniform)
                y_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=2, uniform=uniform
                )
                imec = TIMEC(x_marginal, y_ar_marginal, merge=merge)
                assert_is_distribution(self, imec)

    def test_timec_marginalizes(self):
        for uniform in [True, False]:
            for merge in [True, False]:
                x_marginal = DummyTabularMarginal(num_states=5, uniform=uniform)
                y_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=2, uniform=uniform
                )
                imec = TIMEC(x_marginal, y_ar_marginal, merge=merge)
                assert_marginalizes(self, imec, x_marginal, y_ar_marginal)

    def test_timec_consistent(self):
        for uniform in [True, False]:
            for merge in [True, False]:
                x_marginal = DummyTabularMarginal(num_states=5, uniform=uniform)
                y_ar_marginal = DummyAutoRegressiveMarginal(
                    num_states_per_node=4, depth=2, uniform=uniform
                )
                imec = TIMEC(x_marginal, y_ar_marginal, merge=merge)
                assert_consistency(self, imec, x_marginal, y_ar_marginal)

    def test_timec_raises(self):
        x_marginal = DummyTabularMarginal(num_states=5, uniform=True)
        y_ar_marginal = DummyAutoRegressiveMarginal(
            num_states_per_node=4, depth=2, uniform=True
        )
        with self.assertRaises(TypeError):
            TIMEC(y_ar_marginal, y_ar_marginal)
        with self.assertRaises(TypeError):
            TIMEC(x_marginal, x_marginal)
        with self.assertRaises(TypeError):
            TIMEC(x_marginal, y_ar_marginal, merge="invalid")
        imec = TIMEC(x_marginal, y_ar_marginal)
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

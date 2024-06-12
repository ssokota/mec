from typing import Callable
import unittest

import numpy as np

from mec import ARIMEC, FIMEC, TIMEC
from mec.iterative.algorithms import IMEC
from mec.iterative.marginals import is_probability_zero

from marginals import (
    DummyAutoRegressiveMarginal,
    DummyFactoredMarginal,
    DummyMarginal,
    DummyTabularMarginal,
    TDummy,
    TDummy_F_or_AR,
)


def assert_marginalizes(
    test_case: unittest.TestCase,
    imec: IMEC,
    mu: DummyMarginal,
    nu: DummyMarginal,
) -> None:
    for x in mu.enumerate():
        assert_marginalizes_helper(test_case, imec.evaluate_x_given_y, x, mu, nu)
    for y in nu.enumerate():
        assert_marginalizes_helper(test_case, imec.evaluate_y_given_x, y, nu, mu)


def assert_marginalizes_helper(
    test_case: unittest.TestCase,
    evaluate_a_given_b: Callable[[list[int], list[int]], float],
    a: list[int],
    a_marginal: DummyMarginal,
    b_marginal: DummyMarginal,
) -> None:
    prob1 = np.exp(a_marginal.evaluate(a))
    prob2 = 0.0
    for b in b_marginal.enumerate():
        if not is_probability_zero(b_marginal, b):
            prob2 += np.exp(b_marginal.evaluate(b) + evaluate_a_given_b(a, b))
    test_case.assertTrue(np.isclose(prob1, prob2))


def assert_generalization(
    test_case: unittest.TestCase,
    y_ar_marginal: DummyAutoRegressiveMarginal,
    x_marginal1: DummyTabularMarginal,
    x_marginal2: TDummy_F_or_AR,
) -> None:
    # Check that supports are consistent
    x1_support = set([tuple(x1) for x1 in x_marginal1.enumerate()])
    x2_support = set([tuple(x2) for x2 in x_marginal2.enumerate()])
    test_case.assertEqual(x1_support, x2_support)
    for x in x_marginal1.enumerate():
        test_case.assertEqual(x_marginal1.evaluate(x), x_marginal2.evaluate(x))

    # Initialize appropriate IMECs
    imec = TIMEC(x_marginal1, y_ar_marginal)
    imec_g: FIMEC | ARIMEC
    if isinstance(x_marginal2, DummyFactoredMarginal):
        assert isinstance(x_marginal2, DummyFactoredMarginal)
        imec_g = FIMEC(x_marginal2, y_ar_marginal)
    else:
        assert isinstance(x_marginal2, DummyAutoRegressiveMarginal)
        imec_g = ARIMEC(x_marginal2, y_ar_marginal)

    # Check that X | Y behavior is consistent
    for y in y_ar_marginal.enumerate():
        if not is_probability_zero(y_ar_marginal, y):
            for x in x_marginal1.enumerate():
                test_case.assertTrue(
                    np.isclose(
                        imec.evaluate_x_given_y(x, y), imec_g.evaluate_x_given_y(x, y)
                    )
                )
            est1, ll1 = imec.estimate_x_given_y(y)
            est2, ll2 = imec_g.estimate_x_given_y(y)
            assert tuple(est1) == tuple(est2)
            assert np.isclose(ll1, ll2)

    # Check that Y | X behavior is consistent
    for x in x_marginal1.enumerate():
        if not is_probability_zero(x_marginal1, x):
            for y in y_ar_marginal.enumerate():
                test_case.assertTrue(
                    np.isclose(
                        imec.evaluate_y_given_x(y, x), imec_g.evaluate_y_given_x(y, x)
                    )
                )
            est1, ll1 = imec.estimate_y_given_x(x)
            est2, ll2 = imec_g.estimate_y_given_x(x)
            test_case.assertEqual(tuple(est1), tuple(est2))
            test_case.assertTrue(np.isclose(ll1, ll2))


def assert_consistency(
    test_case: unittest.TestCase,
    imec: IMEC,
    mu: TDummy,
    nu: DummyAutoRegressiveMarginal,
) -> None:
    # Assert p(y|x)p(x)=p(x|y)p(y)
    for x in mu.enumerate():
        for y in nu.enumerate():
            if (not is_probability_zero(imec.mu, x)) and (
                not is_probability_zero(imec.nu, y)
            ):
                x_ll = mu.evaluate(x)
                y_ll = nu.evaluate(y)
                yx_ll = imec.evaluate_y_given_x(y, x)
                xy_ll = imec.evaluate_x_given_y(x, y)
                test_case.assertTrue(
                    np.isclose(np.exp(x_ll + yx_ll), np.exp(y_ll + xy_ll))
                )
            elif is_probability_zero(imec.mu, x) and (
                not is_probability_zero(imec.nu, y)
            ):
                test_case.assertTrue(
                    np.isclose(np.exp(imec.evaluate_x_given_y(x, y)), 0)
                )
            elif is_probability_zero(imec.nu, y) and (
                not is_probability_zero(imec.mu, x)
            ):
                test_case.assertTrue(
                    np.isclose(np.exp(imec.evaluate_y_given_x(y, x)), 0)
                )
            else:
                test_case.assertTrue(
                    is_probability_zero(imec.mu, x) and is_probability_zero(imec.nu, y)
                )

    # Assert estimation and evaluation given consistent likelihoods
    for x in mu.enumerate():
        if not is_probability_zero(mu, x):
            y, ll1 = imec.estimate_y_given_x(x)
            ll2 = imec.evaluate_y_given_x(y, x)
            test_case.assertTrue(np.isclose(ll1, ll2))
    for y in nu.enumerate():
        if not is_probability_zero(nu, y):
            x, ll1 = imec.estimate_x_given_y(y)
            ll2 = imec.evaluate_x_given_y(x, y)
            test_case.assertTrue(np.isclose(ll1, ll2))

    # Assert sampling and evaluation give consistent likelihoods for random samples
    (x, y), ll1 = imec.sample()
    ll2 = imec.evaluate(x, y)
    test_case.assertTrue(np.isclose(ll1, ll2))
    for x in mu.enumerate():
        if not is_probability_zero(mu, x):
            y, ll1 = imec.sample_y_given_x(x)
            ll2 = imec.evaluate_y_given_x(y, x)
            test_case.assertTrue(np.isclose(ll1, ll2))
    for y in nu.enumerate():
        if not is_probability_zero(nu, y):
            x, ll1 = imec.sample_x_given_y(y)
            ll2 = imec.evaluate_x_given_y(x, y)
            test_case.assertTrue(np.isclose(ll1, ll2))

    if isinstance(mu, DummyAutoRegressiveMarginal):
        # Estimation not guaranteed to be consistent with MAP for autoregressive
        return

    # Assert brute force agrees with estimation
    for y in nu.enumerate():
        if is_probability_zero(nu, y):
            continue
        _, x_max_ll = imec.estimate_x_given_y(y)
        for x in mu.enumerate():
            x_ll = imec.evaluate_x_given_y(x, y)
            test_case.assertGreaterEqual(x_max_ll, x_ll)


def assert_is_distribution(test_case: unittest.TestCase, imec: IMEC) -> None:
    assert isinstance(imec.nu, DummyAutoRegressiveMarginal)
    probability_mass = 0
    for x in imec.mu.enumerate():
        for y in imec.nu.enumerate():
            ll = imec.evaluate(x, y)
            test_case.assertLessEqual(ll, 0)
            probability_mass += np.exp(ll)
    test_case.assertTrue(np.isclose(probability_mass, 1))


def run_methods(imec: IMEC) -> None:
    assert isinstance(imec.nu, DummyAutoRegressiveMarginal)
    imec.sample()
    for x in imec.mu.enumerate():
        for y in imec.nu.enumerate():
            imec.evaluate(x, y)
            if not is_probability_zero(imec.mu, x):
                imec.estimate_y_given_x(x)
                imec.sample_y_given_x(x)
                imec.evaluate_y_given_x(y, x)
            if not is_probability_zero(imec.nu, y):
                imec.estimate_x_given_y(y)
                imec.sample_x_given_y(y)
                imec.evaluate_x_given_y(x, y)

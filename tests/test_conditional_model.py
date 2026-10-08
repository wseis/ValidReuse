"""Conditional LRV candidates: algebraic nesting, validation and a smoke fit."""
import unittest
import warnings

import numpy as np

from count_models import (SLOPE_MODES, base_family, conditional_effective_parameters,
                          fit_count_model, population_quantiles, validate_payload)

FAMILIES = ("poisson_lognormal", "negative_binomial")


def paired_payload(n=12, seed=5, **overrides):
    """One influent concentration per effluent sample, true slope 1."""
    rng = np.random.default_rng(seed)
    exposure = 1875. / 100.
    log_inflow = rng.normal(6.8, .45, n)
    log_effluent = (log_inflow - 7.) + rng.normal(0, .30, n)
    payload = dict(vals_zu=list(10. ** log_inflow),
                   vals_ab=list(rng.poisson(exposure * 10. ** log_effluent)),
                   exposures=[exposure] * n, sample_groups=list(range(n)),
                   q=10, alpha=.05, draws=300, warmup=300, chains=2, seed=11)
    payload.update(overrides)
    return payload


class ConditionalAlgebra(unittest.TestCase):
    """The effective-parameter mapping must nest both existing estimands."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.mu = rng.normal(15., 1., 400)
        self.sd = rng.gamma(2., .2, 400)
        self.intercept = rng.normal(0., 1., 400)
        self.spread = rng.gamma(2., .3, 400)
        self.centre = 14.

    def test_zero_slope_reproduces_independent_model(self):
        for family in FAMILIES:
            mu, sd, out = conditional_effective_parameters(
                self.mu, self.sd, self.intercept, 0., self.centre)
            np.testing.assert_allclose(
                population_quantiles(mu, sd, out, self.spread, family, .1),
                population_quantiles(self.mu, self.sd, self.intercept, self.spread, family, .1))

    def test_unit_slope_removes_influent_spread(self):
        mu, sd, _ = conditional_effective_parameters(
            self.mu, self.sd, self.intercept, 1., self.centre)
        np.testing.assert_allclose(sd, 0.)
        np.testing.assert_allclose(mu, self.centre)

    def test_slope_above_one_keeps_spread_positive(self):
        _, sd, _ = conditional_effective_parameters(
            self.mu, self.sd, self.intercept, 1.4, self.centre)
        self.assertTrue(np.all(sd > 0.))

    def test_base_family_strips_prefix(self):
        self.assertEqual(base_family("conditional_negative_binomial"), "negative_binomial")
        self.assertEqual(base_family("poisson_lognormal"), "poisson_lognormal")


class ConditionalValidation(unittest.TestCase):

    def test_requires_one_influent_value_per_effluent_sample(self):
        payload = paired_payload(n=12)
        payload["vals_zu"] = payload["vals_zu"][:5]
        payload["family"] = "conditional_poisson_lognormal"
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_independent_families_keep_free_influent_length(self):
        payload = paired_payload(n=12)
        payload["vals_zu"] = payload["vals_zu"][:5]
        payload["family"] = "poisson_lognormal"
        validate_payload(payload)

    def test_rejects_unknown_slope_mode(self):
        payload = paired_payload(family="conditional_poisson_lognormal", slope_mode="auto")
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_known_slope_modes(self):
        self.assertEqual(set(SLOPE_MODES), {"free", "unit", "zero"})


class ConditionalFit(unittest.TestCase):
    """A short chain: the pinned slopes must bracket the free estimate."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        payload = paired_payload()
        cls.results = {
            mode: fit_count_model(dict(payload, family="conditional_poisson_lognormal",
                                       slope_mode=mode))
            for mode in ("zero", "free", "unit")}
        cls.independent = fit_count_model(dict(payload, family="poisson_lognormal"))

    def test_zero_slope_matches_independent_candidate(self):
        self.assertAlmostEqual(self.results["zero"]["L_alpha"],
                               self.independent["L_alpha"], places=6)

    def test_pinned_slopes_bracket_the_free_estimate(self):
        self.assertLess(self.results["zero"]["median"], self.results["free"]["median"])
        self.assertLessEqual(self.results["free"]["median"], self.results["unit"]["median"] + .2)

    def test_slope_summary_reports_posterior(self):
        summary = self.results["free"]["slope"]
        self.assertEqual(summary["mode"], "free")
        self.assertIsNone(summary["fixed"])
        self.assertLess(summary["low"], summary["high"])
        self.assertEqual(self.results["unit"]["slope"]["fixed"], 1.)

    def test_estimand_is_labelled_separately(self):
        self.assertEqual(self.results["free"]["estimand"], "latent_conditional_population_lrv")
        self.assertEqual(self.independent["estimand"], "latent_independent_population_lrv")
        self.assertIsNone(self.independent["slope"])


if __name__ == "__main__":
    unittest.main()

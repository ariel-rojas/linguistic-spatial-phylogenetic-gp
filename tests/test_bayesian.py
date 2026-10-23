"""Oráculos de la representación GP y del límite de información train/test."""

import importlib.util
import unittest

import numpy as np
from numpy.testing import assert_allclose

from gaussian_processes.bayesian.design import (
    CLASS_CONTRASTS,
    basis_scales,
    phylogenetic_design,
    spherical_basis,
)
from gaussian_processes.models.features import coordinates_xyz
from gaussian_processes.models.spherical import SphericalSplineKernel


class BayesianDesignTests(unittest.TestCase):
    def test_basis_is_exact_existing_kernel(self):
        latitude = np.array([90, -90, 0, 45, -30])
        longitude = np.array([0, 180, 179, -179, 45])
        xyz = coordinates_xyz(latitude, longitude)
        basis, degree = spherical_basis(latitude, longitude, 8)
        for smoothing in [1e-5, 0.001, 0.01]:
            weighted = basis * basis_scales(degree, smoothing)
            assert_allclose(
                weighted @ weighted.T,
                SphericalSplineKernel(8, smoothing)(xyz),
                atol=1e-12,
            )

    def test_symmetric_class_prior(self):
        assert_allclose(CLASS_CONTRASTS.sum(axis=0), 0, atol=1e-14)
        expected = np.full((3, 3), -0.5)
        np.fill_diagonal(expected, 1)
        assert_allclose(CLASS_CONTRASTS @ CLASS_CONTRASTS.T, expected)

    def test_conditional_covariance_and_duplicate_tips(self):
        c = np.array([[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1.0]])
        factor, mean, residual, unique = phylogenetic_design(
            c, np.array([0, 0]), np.array([1, 2])
        )
        self.assertEqual(unique.tolist(), [0])
        assert_allclose(factor @ factor.T, np.ones((2, 2)))
        assert_allclose(mean @ mean.T + residual @ residual.T, np.eye(2))
        assert_allclose(mean @ factor.T, [[0.8, 0.8], [0, 0]])
        assert_allclose(residual @ residual.T, [[0.36, 0], [0, 1]])

    @unittest.skipUnless(importlib.util.find_spec("numpyro"), "Requiere extra bayesian")
    def test_model_likelihood_only_contains_supplied_labels(self):
        import jax
        from numpyro.handlers import seed, trace

        from gaussian_processes.bayesian.model import categorical_gp
        from gaussian_processes.bayesian.workflow import load_sampling_settings

        settings = load_sampling_settings("config/bayesian.json")
        b, d = spherical_basis([0, 1, 2], [0, 1, 2], 2)
        traced = trace(seed(categorical_gp, jax.random.PRNGKey(2))).get_trace(
            b, d, np.eye(3), settings, np.array([0, 1, 2])
        )
        self.assertTrue(traced["observed"]["is_observed"])
        self.assertEqual(traced["observed"]["value"].shape, (3,))
        self.assertEqual(traced["z_spatial"]["value"].shape, (9, 2))
        from gaussian_processes.bayesian.inference import probability_draws

        samples = {
            key: np.asarray(traced[key]["value"])[None, ...]
            for key in [
                "sigma_spatial",
                "sigma_phylogenetic",
                "smoothing",
                "intercept",
                "z_spatial",
                "z_phylogenetic",
            ]
        }
        predicted = probability_draws(samples, b, d, np.eye(3), np.empty((3, 0)), 1)
        assert_allclose(predicted[0], traced["observed"]["fn"].probs, atol=1e-12)
        from jax.experimental.sparse import BCOO

        sparse_trace = trace(seed(categorical_gp, jax.random.PRNGKey(2))).get_trace(
            b, d, BCOO.fromdense(np.eye(3)), settings, np.array([0, 1, 2])
        )
        assert_allclose(
            sparse_trace["observed"]["fn"].probs,
            traced["observed"]["fn"].probs,
            atol=1e-12,
        )

    @unittest.skipUnless(importlib.util.find_spec("numpyro"), "Requiere extra bayesian")
    def test_probabilities_average_after_softmax(self):
        import pandas as pd

        from gaussian_processes.bayesian.workflow import summarize_predictions

        draws = np.array([[[0.8, 0.05, 0.15]], [[0.4, 0.1, 0.5]]])
        rows = summarize_predictions(pd.DataFrame({"ID": ["a"]}), draws)
        assert_allclose(rows[["p_OV", "p_SD", "p_VO"]], [[0.6, 0.075, 0.325]])
        self.assertLess(rows.p_OV_lower.iloc[0], rows.p_OV.iloc[0])
        self.assertGreater(rows.epistemic_information.iloc[0], 0)


if __name__ == "__main__":
    unittest.main()

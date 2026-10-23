"""Los histogramas representan muestras; las barras predictivas, probabilidades."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.stats import lognorm, norm

from gaussian_processes.visualization.bayesian_context import prior_summary
from gaussian_processes.visualization.posterior import (
    build_posterior_payload,
    distribution_summary,
)


class PosteriorViewTests(unittest.TestCase):
    def test_prior_ranges_are_not_truncated_or_renormalized(self):
        normal = prior_summary(norm(scale=1.5), "real", "normal")
        assert_allclose(normal["interval"], [-2.93994598, 2.93994598])
        assert_allclose(sum(normal["chart"]["mass"]), 0.998)
        log = prior_summary(lognorm(s=1.5, scale=0.001), "positive", "lognormal", True)
        assert_allclose(log["median"], 0.001)
        assert_allclose(log["chart"]["mean"], -3)
        assert_allclose(10 ** np.array(log["chart"]["interval"]), log["interval"])
        assert_allclose(sum(log["chart"]["mass"]), 0.998)

    def test_histogram_conserves_mass_and_endpoints(self):
        result = distribution_summary([0, 0.2, 0.8, 1], np.linspace(0, 1, 21))
        assert_allclose(sum(result["mass"]), 1)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["mean"], 0.5)
        assert_allclose(result["interval"], [0.015, 0.985])

    def test_constant_draws_and_invalid_input(self):
        result = distribution_summary(np.ones((4, 100)))
        self.assertEqual(result["mean"], 1)
        self.assertEqual(result["interval"], [1, 1])
        for values in [[], [np.nan], [-0.1, 0.2]]:
            with self.assertRaises(ValueError):
                distribution_summary(values, np.linspace(0, 1, 21))

    def test_payload_aligns_ids_and_rejects_stale_predictions(self):
        probabilities = np.array([[[0.8, 0.1, 0.1]], [[0.4, 0.2, 0.4]]])
        rows = pd.DataFrame(
            {
                "ID": ["a", "b"],
                "Name": ["A", "B"],
                "familia_low": ["F", "F"],
                "role": ["test", "imputed"],
                "observado": ["OV", "Faltante"],
                "predicho": ["OV", "OV"],
                "p_OV": [0.6, 0.6],
                "p_SD": [0.15, 0.15],
                "p_VO": [0.25, 0.25],
            }
        )
        provenance = {
            "sampling": {},
            "evaluation_diagnostics": {},
            "imputation_diagnostics": {},
            "status": "complete",
        }
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for stage, identifier in [("evaluation", "a"), ("imputation", "b")]:
                np.savez(
                    folder / f"{stage}.npz",
                    classes=["OV", "Sin dominante", "VO"],
                    target_ids=[identifier],
                    fit_ids=["train"],
                    probabilities=probabilities,
                    settings_json=json.dumps({}),
                    diagnostics_json=json.dumps({}),
                    posterior_sigma_spatial=np.ones((2, 4)),
                    posterior_sigma_phylogenetic=np.ones((2, 4)),
                    posterior_smoothing=np.ones((2, 4)) * 0.001,
                    posterior_intercept=np.zeros((2, 4, 2)),
                )
            payload = build_posterior_payload(folder, provenance, rows)
            self.assertEqual(
                [c["stage"] for c in payload["cases"]], ["evaluation", "imputation"]
            )
            self.assertEqual(
                payload["cases"][0]["distributions"][0]["mean"], 0.6000000000000001
            )
            rows.loc[0, "p_OV"] = 0.9
            with self.assertRaisesRegex(ValueError, "Medias"):
                build_posterior_payload(folder, provenance, rows)


if __name__ == "__main__":
    unittest.main()

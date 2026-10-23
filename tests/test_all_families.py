"""Comprueba que la agregación por familia no se confunde con la de lenguas."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gaussian_processes.configuration import load_configuration
from gaussian_processes.evaluation.all_families import evaluate_all_families, summarize


class FamilyTests(unittest.TestCase):
    def test_each_family_is_predicted_once(self):
        """Ejecuta el recorrido con familias sintéticas; una solo contiene OV."""
        labels = ["OV", "VO", "Sin dominante"] * 2 + ["OV"] * 3
        data = pd.DataFrame(
            {
                "ID": [f"id{i}" for i in range(9)],
                "unidad": [f"id{i}" for i in range(9)],
                "observado": labels,
                "elegible_modelo": True,
                "familia_id": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
                "tree_tip_index": list(range(9)),
                "Latitude": [0.0] * 9,
                "Longitude": list(range(9)),
            }
        )
        config, _ = load_configuration()
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=1):
            folder = Path(temporary)
            with (
                patch(
                    "gaussian_processes.evaluation.all_families.load_data",
                    return_value=data,
                ),
                patch(
                    "gaussian_processes.evaluation.all_families.load_configuration",
                    return_value=(config, folder),
                ),
                patch("gaussian_processes.evaluation.all_families.plot_distribution"),
                patch(
                    "gaussian_processes.evaluation.all_families.load_phylogeny",
                    return_value=(np.eye(9) * 10, {}),
                ),
            ):
                evaluate_all_families("unused.json")
            predictions = pd.read_csv(folder / "families/predictions.csv")
            self.assertTrue(predictions.ID.is_unique)
            self.assertEqual(set(predictions.ID), set(data.ID))
            self.assertTrue(
                (predictions.familia_id == predictions.held_out_family).all()
            )

    def test_family_and_language_means_differ(self):
        metrics = pd.DataFrame(
            {
                "accuracy": [1.0, 0.0],
                "balanced_accuracy": [1.0, 0.0],
                "macro_f1": [1 / 3, 0.0],
                "log_loss": [0.2, 2.0],
                "brier_multiclass": [0.06, 1.46],
            }
        )
        predictions = pd.DataFrame(
            {
                "observado": ["VO"] + ["OV"] * 9,
                "p_OV": [0.1] * 10,
                "p_SD": [0.1] * 10,
                "p_VO": [0.8] * 10,
            }
        )
        summary, _ = summarize(metrics, predictions)
        self.assertEqual(summary["mean_by_family"]["accuracy"], 0.5)
        self.assertEqual(summary["pooled_by_language"]["accuracy"], 0.1)

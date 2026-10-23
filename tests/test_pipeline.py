"""Pruebas independientes de resultados históricos: geometría, métricas y particiones."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from gaussian_processes.configuration import (
    CLASSES,
    PROBABILITY_COLUMNS,
    load_configuration,
)
from gaussian_processes.data.loading import load_data
from gaussian_processes.evaluation.metrics import classification_metrics
from gaussian_processes.evaluation.search import select_parameters
from gaussian_processes.evaluation.splits import inner_splits, split_data
from gaussian_processes.models.features import coordinates_xyz
from gaussian_processes.models.gp import predict
from gaussian_processes.models.kernels import build_kernel
from gaussian_processes.visualization.map import imputation_view


def sample_data():
    """Muestra pequeña con tres clases, tres familias y grupos repetidos."""
    n = 36
    return pd.DataFrame(
        {
            "ID": [f"id{i}" for i in range(n)],
            "unidad": [f"g{i // 2}" for i in range(n)],
            "familia_id": [f"f{i // 12}" for i in range(n)],
            "observado": [CLASSES[(i // 2) % 3] for i in range(n)],
        }
    )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config, _ = load_configuration("config/multiclass.json")

    def test_spherical_geometry(self):
        xyz = coordinates_xyz(np.array([0, 0, 90, 90]), np.array([179, -179, 0, 140]))
        self.assertLess(np.linalg.norm(xyz[0] - xyz[1]), 223)
        assert_allclose(xyz[2], xyz[3], atol=1e-9)

    def test_native_kernel_and_covariance(self):
        from sklearn.gaussian_process.kernels import Matern

        kernel = build_kernel(self.config, np.array([[10.0, 8.0], [8.0, 10.0]]))
        self.assertIsInstance(kernel.k1.k2.kernel, Matern)
        from gaussian_processes.models.phylogenetic import PhylogeneticKernel

        self.assertIsInstance(kernel.k2.k2.kernel, PhylogeneticKernel)
        self.assertEqual(kernel.k1.k2.kernel.nu, 0.5)
        X = np.array([[0, 0, 0, 0], [1500, 0, 0, 1]], float)
        K = clone(kernel)(X)
        self.assertAlmostEqual(
            K[0, 1], np.exp(-1) + 8 * self.config["phylogenetic_rate"]
        )
        assert_allclose(K.diagonal(), kernel.diag(X))
        self.assertGreater(np.linalg.eigvalsh(K).min(), 0)
        self.assertEqual(kernel.theta.size, 0)

    def test_nested_groups_are_disjoint(self):
        df = sample_data()
        train, test = split_data(df, self.config)
        self.assertFalse(set(df.iloc[train].unidad) & set(df.iloc[test].unidad))
        outer_train = df.iloc[train]
        for fit, validation in inner_splits(outer_train, self.config):
            self.assertFalse(
                set(outer_train.iloc[fit].unidad)
                & set(outer_train.iloc[validation].unidad)
            )
            self.assertFalse(
                set(outer_train.iloc[validation].ID) & set(df.iloc[test].ID)
            )

    def test_family_split_and_conflicting_identity(self):
        df = sample_data()
        config = dict(self.config, test_family="f2")
        train, test = split_data(df, config, "family")
        self.assertTrue(df.iloc[test].familia_id.eq("f2").all())
        self.assertFalse(df.iloc[train].familia_id.eq("f2").any())
        df.loc[24, "unidad"] = df.loc[0, "unidad"]
        with self.assertRaises(ValueError):
            split_data(df, config, "family")

    def test_metrics_hand_calculation(self):
        prob = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.7, 0.2]])
        metrics = classification_metrics(CLASSES, prob)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["log_loss"], -np.log([0.8, 0.6, 0.2]).mean())
        self.assertAlmostEqual(metrics["brier_multiclass"], (0.06 + 0.24 + 1.14) / 3)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 1, 0]])
        self.assertAlmostEqual(metrics["macro_f1"], (1 + 2 / 3) / 3)

    def test_absent_class_is_not_perfect_recall(self):
        metrics = classification_metrics(["OV"], [[0.8, 0.1, 0.1]])
        self.assertIsNone(metrics["per_class"]["Sin dominante"]["recall"])
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)
        self.assertEqual(metrics["balanced_accuracy"], 1)

    def test_grid_refits_selected_parameters(self):
        df = sample_data()
        X = np.column_stack(
            [
                np.arange(len(df)),
                np.zeros((len(df), 2)),
                np.arange(len(df)),
            ]
        )
        config = dict(
            self.config,
            grid={
                "length_scale_km": [500, 1500],
                "spatial_variance": [1],
                "phylogenetic_rate": [0.002, 0.008],
                "nu": [0.5, 1.5],
            },
        )
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=1):
            model, selected, _ = select_parameters(
                df, X, config, Path(temporary), np.eye(len(df)) * 10
            )
            results = pd.read_csv(Path(temporary) / "grid_results.csv")
            self.assertEqual(len(results), 8)
            best = results.iloc[0]
            for key in config["grid"]:
                self.assertEqual(selected[key], best[key])
            for estimator in model.base_estimator_.estimators_:
                self.assertEqual(
                    estimator.kernel_.k1.k2.kernel.length_scale,
                    selected["length_scale_km"],
                )
                self.assertEqual(estimator.kernel_.k1.k2.kernel.nu, selected["nu"])
            prob, labels = predict(model, X)
            assert_allclose(prob.sum(axis=1), 1)
            self.assertEqual(list(model.classes_), CLASSES)
            self.assertEqual(len(labels), len(df))

    def test_default_grid_has_108_unique_combinations(self):
        from sklearn.model_selection import ParameterGrid

        self.assertEqual(len(ParameterGrid(self.config["grid"])), 108)

    def test_map_keeps_excluded_rows_and_three_probabilities(self):
        observations = pd.DataFrame(
            {"ID": ["a", "b", "c"], "observado": ["OV", "Faltante", "Faltante"]}
        )
        imputed = pd.DataFrame(
            {
                "ID": ["b"],
                "predicho": ["Sin dominante"],
                "p_OV": [0.1],
                "p_SD": [0.8],
                "p_VO": [0.1],
            }
        )
        view = imputation_view(observations, imputed).set_index("ID")
        self.assertEqual(view.loc["b", "predicho"], "Sin dominante")
        self.assertEqual(view.loc["c", "estado"], "Faltante sin imputar")
        self.assertTrue(set(PROBABILITY_COLUMNS) <= set(view.columns))

    def test_real_data_split_has_no_identity_leakage(self):
        data = load_data()
        known = data.loc[data.elegible_modelo & data.observado.isin(CLASSES)]
        train, test = split_data(known, self.config)
        self.assertGreater(len(known), 1000)
        for column in [
            "unidad",
            "Glottocode",
            "ISO639P3code",
            "low_iso",
            "tree_glottocode",
        ]:
            self.assertFalse(
                (set(known.iloc[train][column]) - {""})
                & (set(known.iloc[test][column]) - {""})
            )
        self.assertEqual(set(known.iloc[test].observado), set(CLASSES))
        # La estratificación aproxima las frecuencias sin dividir identidades.
        train_rates = known.iloc[train].observado.value_counts(normalize=True)
        test_rates = known.iloc[test].observado.value_counts(normalize=True)
        assert_allclose(
            train_rates.reindex(CLASSES), test_rates.reindex(CLASSES), atol=0.02
        )
        repeated_train, repeated_test = split_data(known, self.config)
        assert_allclose(train, repeated_train)
        assert_allclose(test, repeated_test)
        tasmanian = data.set_index("ID").loc["toy"]
        self.assertFalse(tasmanian.elegible_modelo)


if __name__ == "__main__":
    unittest.main()

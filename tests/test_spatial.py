"""Pruebas de validez, integración sklearn y separación geográfica."""

import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from shapely.geometry import box
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from gaussian_processes.configuration import CLASSES, load_configuration
from gaussian_processes.data.geography import classify_locations
from gaussian_processes.evaluation.search import select_parameters
from gaussian_processes.evaluation.spatial import spatial_splits
from gaussian_processes.models.features import coordinates_xyz
from gaussian_processes.models.kernels import build_kernel
from gaussian_processes.models.spherical import (
    EARTH_RADIUS_KM,
    GeodesicExponentialKernel,
    SphericalSplineKernel,
    great_circle_distance,
)


class SpatialTests(unittest.TestCase):
    def test_configuration_rejects_parameters_of_another_strategy(self):
        config, _ = load_configuration("config/geodesic_exponential.json")
        config["nu"] = 1.5
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_configuration(path)
            config, _ = load_configuration()
            self.assertEqual(config["spatial_kernel"], "spherical_spline")
            config["grid"]["max_degree"] = [8.5]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_configuration(path)

    def test_surface_geometry_extremes(self):
        xyz = coordinates_xyz([0, 0, 0, 90, 90], [0, 180, 2, -170, 20])
        distances = great_circle_distance(xyz)
        assert_allclose(distances.diagonal(), 0, atol=1e-10)
        self.assertAlmostEqual(distances[0, 1], np.pi * EARTH_RADIUS_KM)
        self.assertAlmostEqual(distances[0, 2], np.radians(2) * EARTH_RADIUS_KM)
        self.assertLess(distances[3, 4], 1e-9)
        seam = coordinates_xyz([0, 0], [179, -179])
        self.assertAlmostEqual(great_circle_distance(seam)[0, 1], distances[0, 2])
        with self.assertRaises(ValueError):
            great_circle_distance([[0, 0, 0]])

    def test_kernels_psd_rotation_and_rectangular_blocks(self):
        rng = np.random.default_rng(42)
        xyz = rng.normal(size=(60, 3))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        for kernel in (
            GeodesicExponentialKernel(3000),
            SphericalSplineKernel(16, 0.001),
        ):
            K, gradient = clone(kernel)(xyz, eval_gradient=True)
            assert_allclose(K, K.T, atol=1e-12)
            assert_allclose(K.diagonal(), kernel.diag(xyz), atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(K).min(), -1e-10)
            assert_allclose(K, kernel(xyz @ rotation), atol=1e-12)
            assert_allclose(K[:8, 8:], kernel(xyz[:8], xyz[8:]), atol=1e-12)
            self.assertEqual(gradient.shape, (60, 60, 0))
            self.assertEqual(kernel.theta.size, 0)

    def test_spline_degree_one_matches_analytic_basis(self):
        # P0=1 y P1=cos(theta); los tres armónicos l=1 suman 3*cos(theta).
        xyz = coordinates_xyz([0, 0, 90], [0, 180, 0])
        smoothing = 0.5
        cosine = (xyz / EARTH_RADIUS_KM) @ (xyz / EARTH_RADIUS_KM).T
        weight = 3 / (1 + 4 * smoothing)
        expected = (1 + weight * cosine) / (1 + weight)
        assert_allclose(SphericalSplineKernel(1, smoothing)(xyz), expected, atol=1e-12)

    def test_new_grid_searches_and_phylogeny_unchanged(self):
        n = 45
        data = pd.DataFrame(
            {"unidad": [f"u{i}" for i in range(n)], "observado": CLASSES * 15}
        )
        X = np.column_stack(
            [
                coordinates_xyz(np.linspace(-50, 50, n), np.linspace(-170, 170, n)),
                np.arange(n),
            ]
        )
        covariance = np.eye(n) * 10 + 2
        original = covariance.copy()
        for filename in ("geodesic_exponential", "spherical_spline"):
            config, _ = load_configuration(f"config/{filename}.json")
            from sklearn.model_selection import ParameterGrid

            self.assertEqual(len(ParameterGrid(config["grid"])), 108)
            config["grid"] = {key: values[:1] for key, values in config["grid"].items()}
            config["grid"]["spatial_variance"] = [0.25, 1]
            kernel = build_kernel(config, covariance)
            assert_allclose(kernel.k2(X), config["phylogenetic_rate"] * covariance)
            with (
                tempfile.TemporaryDirectory() as temporary,
                threadpool_limits(limits=1),
            ):
                model, selected, summary = select_parameters(
                    data, X, config, Path(temporary), covariance
                )
                self.assertEqual(summary["n_candidates"], 2)
                assert_allclose(model.predict_proba(X).sum(axis=1), 1)
                self.assertIn(selected["spatial_variance"], [0.25, 1])
        assert_allclose(covariance, original)

    def test_spatial_blocks_preserve_identity_and_buffer(self):
        rng = np.random.default_rng(18)
        n = 90
        xyz = coordinates_xyz(rng.uniform(-70, 70, n), rng.uniform(-180, 180, n))
        data = pd.DataFrame(
            {"unidad": [f"u{i}" for i in range(n)], "observado": CLASSES * 30}
        )
        # Una identidad aparece en lugares distintos y debe unir ambos bloques.
        data.loc[1, "unidad"] = data.loc[0, "unidad"]
        groups, folds = spatial_splits(data, xyz, 3, 2000, 500)
        self.assertEqual(groups[0], groups[1])
        held_out = []
        for train, test, purged in folds:
            self.assertFalse(set(groups[train]) & set(groups[test]))
            self.assertFalse(set(data.iloc[train].unidad) & set(data.iloc[test].unidad))
            self.assertGreaterEqual(
                great_circle_distance(xyz[train], xyz[test]).min(), 500
            )
            self.assertFalse(set(train) & set(purged))
            held_out.extend(test)
        self.assertEqual(sorted(held_out), list(range(n)))

    def test_geography_is_descriptive_and_retains_missing_coordinates(self):
        land = gpd.GeoDataFrame(geometry=[box(-10, -10, 0, 10)], crs=4326)
        ocean = gpd.GeoDataFrame(geometry=[box(0, -10, 10, 10)], crs=4326)
        mountain = gpd.GeoDataFrame(
            {"mountain_name": ["Cordillera de prueba"]},
            geometry=[box(-8, -5, -2, 5)],
            crs=4326,
        )
        data = pd.DataFrame(
            {
                "ID": ["land", "sea", "boundary", "invalid"],
                "Longitude": [-5, 5, 0, np.nan],
                "Latitude": [0, 0, 0, np.nan],
                "coordenadas_validas": [True, True, True, False],
            }
        )
        result = classify_locations(data, land, ocean, mountain).set_index("ID")
        self.assertEqual(len(result), len(data))
        self.assertEqual(result.loc["land", "mountain_region"], "Cordillera de prueba")
        self.assertIn("Océano", result.loc["sea", "geographic_status"])
        self.assertIn("Límite", result.loc["boundary", "geographic_status"])
        self.assertIn("Sin coordenadas", result.loc["invalid", "geographic_status"])


if __name__ == "__main__":
    unittest.main()

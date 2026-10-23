"""El informe admite pendientes, pero rechaza comparaciones inválidas."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gaussian_processes.evaluation.report import comparison_report


class ReportTests(unittest.TestCase):
    def write_run(self, root, name, digest="same", test_id="language"):
        """Resultado mínimo independiente de los experimentos reales."""
        folder = root / "results" / name
        folder.mkdir(parents=True)
        run = {
            "data_sha256": digest,
            "phylogeny": {"files": {"tree": "same-tree"}},
            "metrics": {
                "accuracy": 0.8, "macro_f1": 0.7, "log_loss": 0.4,
                "per_class": {"Sin dominante": {"recall": 0.2}},
            },
            "search": {
                "n_candidates": 108, "best_mean_validation_log_loss": 0.5,
                "best_parameters": {}, "boundary_parameters": [],
            },
        }
        (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
        (folder / "predictions.csv").write_text(
            f"ID,role\n{test_id},test\n", encoding="utf-8"
        )

    def test_partial_report_names_missing_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_run(root, "spherical_spline")
            with patch("gaussian_processes.evaluation.report.ROOT", root):
                comparison_report()
            report = (root / "results/comparison.md").read_text(encoding="utf-8")
            self.assertIn("Exponencial geodésico: búsqueda sin resultado completo", report)
            self.assertIn("| Spline esférico | 108 |", report)
            self.assertNotIn("| Exponencial geodésico | 108 |", report)

    def test_different_data_or_test_is_rejected(self):
        for digest, test_id in [("different", "language"), ("same", "other")]:
            with self.subTest(digest=digest, test_id=test_id):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self.write_run(root, "spherical_spline")
                    self.write_run(root, "phylogenetic_mcc", digest, test_id)
                    with patch("gaussian_processes.evaluation.report.ROOT", root):
                        with self.assertRaisesRegex(ValueError, "no son comparables"):
                            comparison_report()

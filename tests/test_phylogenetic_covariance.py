"""Verifica escala, separación familiar, alias y aplicación a los datos reales."""

import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from gaussian_processes.configuration import load_configuration
from gaussian_processes.data.loading import load_data
from gaussian_processes.data.phylogeny import align_tip_families, load_phylogeny
from gaussian_processes.models.phylogenetic_covariance import PhylogeneticCovariance


class CovarianceTests(unittest.TestCase):
    def test_normalization_and_blocks(self):
        # Diagonales distintas: dividir por una altura común sería incorrecto.
        source = np.array([[4., 2., 1.], [2., 9., 3.], [1., 3., 16.]])
        original = source.copy()
        result = PhylogeneticCovariance("within_family", True).transform(
            source, ["a", "a", "b"]
        )
        assert_allclose(result, [[1, 1/3, 0], [1/3, 1, 0], [0, 0, 1]])
        assert_allclose(source, original)
        self.assertGreater(np.linalg.eigvalsh(result).min(), 0)
        assert_allclose(PhylogeneticCovariance().transform(source), source)
        assert_allclose(
            PhylogeneticCovariance("global", True).transform(source)[0, 2], 1/8
        )

    def test_aliases_unknowns_and_conflicts(self):
        tips = pd.DataFrame({"tree_glottocode": ["a", "b", "c"]})
        data = pd.DataFrame({"Glottocode": ["a", "a", "b", "c"],
                             "familia_id": ["f", "", "", None]})
        self.assertEqual(align_tip_families(tips, data), ["family:f", "tip:b", "tip:c"])
        # Registros sin hoja no son alias entre sí ni afectan el bloqueo.
        extra = pd.DataFrame({"Glottocode": ["", ""], "familia_id": ["x", "y"]})
        self.assertEqual(align_tip_families(tips, pd.concat([data, extra])),
                         ["family:f", "tip:b", "tip:c"])
        data.loc[1, "familia_id"] = "other"
        with self.assertRaisesRegex(ValueError, "contradictorias"):
            align_tip_families(tips, data)

    def test_invalid_policy_and_variance(self):
        for policy in [PhylogeneticCovariance("bad"), PhylogeneticCovariance("global", 1)]:
            with self.assertRaises(ValueError):
                policy.transform(np.eye(2))
        with self.assertRaises(ValueError):
            PhylogeneticCovariance(normalize=True).transform(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            PhylogeneticCovariance("within_family").transform(np.eye(2), ["", ""])

    def test_real_default_and_equivalent_amplitude(self):
        config, _ = load_configuration()
        data = load_data()
        matrix, metadata = load_phylogeny(config, data)
        assert_allclose(matrix.diagonal(), 1)
        legacy = dict(config, phylogenetic_structure="global", phylogenetic_normalize=False)
        raw, _ = load_phylogeny(legacy, data)
        assert_allclose(config["phylogenetic_rate"] * matrix.diagonal(), .008 * raw.diagonal())
        rows = data.loc[data.tree_tip_index.ge(0) & data.familia_id.ne("")].drop_duplicates("tree_tip_index")
        indices = rows.tree_tip_index.to_numpy()
        families = rows.familia_id.to_numpy()
        different = families[:, None] != families[None, :]
        self.assertTrue(np.all(matrix[np.ix_(indices, indices)][different] == 0))
        self.assertEqual(metadata["model_covariance"]["structure"], "within_family")


if __name__ == "__main__":
    unittest.main()

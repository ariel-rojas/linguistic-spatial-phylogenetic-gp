"""Oráculos pequeños e independientes para tiempos, índices y compatibilidad sklearn."""

import io
import unittest

import numpy as np
from Bio import Phylo
from numpy.testing import assert_allclose
from sklearn.base import clone

from gaussian_processes.configuration import load_configuration
from gaussian_processes.data.loading import load_data
from gaussian_processes.data.phylogeny import (
    ROOT,
    TREE_NAME,
    build_shared_time_covariance,
    load_phylogeny,
    read_tree,
)
from gaussian_processes.models.phylogenetic import PhylogeneticKernel


class PhylogenyTests(unittest.TestCase):
    def test_shared_time_and_root_preserved(self):
        tree = Phylo.read(io.StringIO("((A:2,B:2):8,C:10);"), "newick")
        C = build_shared_time_covariance(tree, ["A", "B", "C"])
        assert_allclose(C, [[10, 8, 0], [8, 10, 0], [0, 0, 10]])
        # Subseleccionar A/B no pierde los ocho milenios hasta su ancestro.
        assert_allclose(build_shared_time_covariance(tree, ["B", "A"]), C[:2, :2])
        tree.get_terminals()[0].branch_length = -1
        with self.assertRaises(ValueError):
            build_shared_time_covariance(tree, ["A"])

    def test_blocks_duplicates_gradient_and_clone(self):
        C = np.array([[10.0, 8.0, 0.0], [8.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        kernel = clone(PhylogeneticKernel(C))
        X, Y = np.array([[2], [0], [0]]), np.array([[1], [2]])
        assert_allclose(kernel(X, Y), [[0, 10], [8, 0], [8, 0]])
        K, gradient = kernel(X, eval_gradient=True)
        assert_allclose(kernel.diag(X), K.diagonal())
        self.assertEqual(gradient.shape, (3, 3, 0))
        self.assertEqual(kernel.theta.size, 0)
        for invalid in [
            np.array([[-1]]),
            np.array([[0.5]]),
            np.array([[3]]),
            np.array([[np.nan]]),
        ]:
            with self.assertRaises(ValueError):
                kernel(invalid)
        with self.assertRaises(ValueError):
            kernel(X, Y, eval_gradient=True)

    def test_family_excludes_unknown_metadata(self):
        import pandas as pd

        from gaussian_processes.evaluation.splits import split_data

        data = pd.DataFrame(
            {"familia_id": ["", "a", "b"], "unidad": ["u0", "u1", "u2"]}
        )
        train, test = split_data(data, {"test_family": "a"}, "family")
        self.assertEqual(train.tolist(), [2])
        self.assertEqual(test.tolist(), [1])

    def test_real_pairs_match_independent_mrca(self):
        config, _ = load_configuration("config/spherical_spline.json")
        data = load_data()
        C, metadata = load_phylogeny(config, data)
        tree = read_tree(ROOT / "datos/originales/phylogeny" / TREE_NAME)
        tips = {tip.name: tip for tip in tree.get_terminals()}
        depths = tree.depths()
        selected = (
            data.loc[data.tree_tip_index.ge(0)]
            .drop_duplicates("tree_tip_index")
            .sample(12, random_state=42)
        )
        for a, b in zip(selected.iloc[:6].itertuples(), selected.iloc[6:].itertuples()):
            ancestor = tree.common_ancestor(tips[a.Glottocode], tips[b.Glottocode])
            self.assertAlmostEqual(
                C[a.tree_tip_index, b.tree_tip_index], depths[ancestor]
            )
        self.assertGreater(metadata["minimum_eigenvalue"], -1e-8)
        self.assertGreater(C.diagonal().min(), 100)  # No se normalizó.
        self.assertTrue((data.elegible_modelo & data.familia_id.eq("")).any())
        self.assertFalse(data.set_index("ID").loc["toy", "elegible_modelo"])


if __name__ == "__main__":
    unittest.main()

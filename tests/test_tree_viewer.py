"""Pruebas de conservación de ramas, etiquetas y metadata del visualizador.

Requieren el extra viewer; la preparación masiva se prueba aparte con archivos reales.
"""

import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("dendropy"), "Instala el extra viewer")
class TreeViewerTests(unittest.TestCase):
    def test_annotations_labels_and_time_are_preserved(self):
        from tree_viewer.readers import parse_tree, serialize_tree

        tree = parse_tree(
            "((101[&height=4.2,location={1,2},custom=hello]:2,102:2)777:3,103:5)999;"
        )
        entry = {"id": "fixture", "years": 3500, "kind": "posterior", "sample": 1}
        table = {
            name: {"tips": name, "unknown_field": "value"}
            for name in ["101", "102", "103"]
        }
        with patch("tree_viewer.readers.language_metadata", return_value={}):
            result = serialize_tree(tree, entry, table)
        self.assertEqual(result["root_age_years"], 8500)
        node = next(n for n in result["nodes"] if n["source_label"] == "101")
        self.assertEqual(node["age_years"], 3500)
        self.assertEqual(node["length"], 2)
        metadata = result["metadata"][node["id"]]
        self.assertEqual(metadata["csv"]["unknown_field"], "value")
        self.assertIn("custom=hello", metadata["annotations"]["source_comment"])
        self.assertEqual(float(metadata["annotations"]["height"]), 4.2)
        internal = next(n for n in result["nodes"] if n["source_label"] == "777")
        self.assertEqual(internal["leaves"], 2)
        self.assertNotIn("posterior", result["metadata"][internal["id"]]["annotations"])

    def test_mismatched_table_and_negative_branches_fail(self):
        from tree_viewer.readers import parse_tree, serialize_tree

        entry = {"years": 0, "kind": "posterior"}
        with patch("tree_viewer.readers.language_metadata", return_value={}):
            with self.assertRaises(ValueError):
                serialize_tree(parse_tree("(a:1,b:1);"), entry, {"c": {}})
            with self.assertRaises(ValueError):
                serialize_tree(parse_tree("(a:-1,b:1);"), entry)

    def test_one_to_many_wals_is_not_assigned_to_ancestors(self):
        from tree_viewer.readers import parse_tree, serialize_tree

        entry = {"years": 0, "kind": "mcc"}
        metadata = {
            "abcd1234": {
                "source_label": "abcd1234_Name_Family",
                "wals": [{"Name": "Name A"}, {"Name": "Name B"}],
            }
        }
        with patch("tree_viewer.readers.language_metadata", return_value=metadata):
            result = serialize_tree(parse_tree("(abcd1234:1,other:1);"), entry)
        leaf = next(n for n in result["nodes"] if n["glottocode"] == "abcd1234")
        self.assertEqual(len(result["metadata"][leaf["id"]]["language"]["wals"]), 2)
        self.assertEqual(result["metadata"][0]["language"], {})

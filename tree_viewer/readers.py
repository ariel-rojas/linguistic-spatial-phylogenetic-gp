"""Lectura fiel de ramas, anotaciones y tablas, sin recalcular la filogenia.

Los ID internos son locales a cada árbol. La edad visual se deriva de longitudes
publicadas; las anotaciones (height, HPD, etc.) se muestran por separado intactas.
"""

import csv
import gzip
import io
import math
import re
from functools import lru_cache

import dendropy
from dendropy.dataio.nexusprocessing import parse_comment_metadata_dendropy_v5_0_0

from .prepare import DATA, ROOT

MCC_PATH = (
    ROOT
    / "datos/originales/phylogeny/global-language-summary-tree-relabelled_cartoonised.tree.gz"
)


@lru_cache(maxsize=1)
def mcc_source():
    """Resuelve Translate por líneas: los nombres publicados contienen paréntesis.

    Leer el bloque Taxlabels literalmente con un parser NEXUS divide esos nombres.
    Extraemos el Newick anotado STATE_1 sin eliminar comentarios ni tocar ramas.
    """
    with gzip.open(MCC_PATH, "rt", encoding="utf-8") as handle:
        text = handle.read()
    translation = re.search(r"(?is)\btranslate\s+(.*?);", text).group(1)
    labels = dict(re.findall(r"(?m)^\s*(\d+)\s+(.+?)\s*,?\s*$", translation))
    labels = {key: value.rstrip(",") for key, value in labels.items()}
    newick = re.search(r"(?is)\btree\s+STATE_1\s*=\s*([^;]+);", text).group(1) + ";"
    return newick, labels


def read_member(reference):
    """Lee un miembro por su offset descomprimido; nunca extrae rutas al disco."""
    with gzip.open(DATA / reference["archive"], "rb") as handle:
        handle.seek(reference["offset"])
        content = handle.read(reference["size"])
    if len(content) != reference["size"]:
        raise ValueError("Archivo incompleto: vuelve a preparar el catálogo")
    return content.decode("utf-8-sig")


def parse_annotations(comment):
    """Conserva tanto los campos interpretados como el comentario literal."""
    return [
        *parse_comment_metadata_dendropy_v5_0_0(comment),
        ("source_comment", comment),
    ]


def parse_tree(text):
    """Los números internos son etiquetas, nunca probabilidades de soporte."""
    return dendropy.Tree.get(
        data=text,
        schema="newick",
        preserve_underscores=True,
        rooting="default-rooted",
        extract_comment_metadata=parse_annotations,
    )


@lru_cache(maxsize=1)
def language_metadata():
    """Asocia nombres originales y todas las filas WALS que comparten glottocode."""
    _, labels = mcc_source()
    metadata = {
        label.split("_", 1)[0]: {"source_label": label, "wals": []}
        for label in labels.values()
    }
    path = ROOT / "datos/lenguas.csv"
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["Glottocode"] in metadata:
                    metadata[row["Glottocode"]]["wals"].append(row)
    return metadata


def serialize_tree(tree, entry, table=None, translations=None):
    """Construye nodos planos y metadata por ID; no duplica el árbol en cada nodo."""
    table = table or {}
    translations = translations or {}
    languages = language_metadata()
    original_nodes = list(tree.preorder_node_iter())
    indices = {node: i for i, node in enumerate(original_nodes)}
    depths, counts = {}, {}
    for node in original_nodes:
        length = float(node.edge_length or 0)
        if not math.isfinite(length) or length < 0:
            raise ValueError("Longitud de rama inválida")
        depths[node] = depths[node.parent_node] + length if node.parent_node else 0.0
    for node in tree.postorder_node_iter():
        counts[node] = sum(counts[child] for child in node.child_node_iter()) or 1
    tip_depths = [depths[node] for node in tree.leaf_node_iter()]
    crown = max(tip_depths) + entry["years"] / 1000
    nodes, metadata = [], {}
    for node in original_nodes:
        index = indices[node]
        raw_label = node.taxon.label if node.taxon else (node.label or "")
        source_label = translations.get(raw_label, raw_label)
        glottocode = (
            source_label.split("_", 1)[0]
            if not entry["years"] and node.is_leaf()
            else ""
        )
        language = languages.get(glottocode, {})
        display_label = language.get("source_label", source_label)
        parts = display_label.split("_", 2)
        name = (parts[1] if len(parts) > 1 else display_label) or f"Clado {index}"
        if language.get("wals"):
            name = language["wals"][0]["Name"]
        nodes.append(
            {
                "id": index,
                "parent": indices.get(node.parent_node),
                "children": [indices[c] for c in node.child_node_iter()],
                "label": name,
                "source_label": source_label,
                "glottocode": glottocode,
                "leaves": counts[node],
                "length": float(node.edge_length or 0),
                "depth": depths[node],
                "age_years": (crown - depths[node]) * 1000,
            }
        )
        metadata[index] = {
            "annotations": node.annotations.values_as_dict(),
            "edge_annotations": node.edge.annotations.values_as_dict(),
            "comments": node.comments,
            "csv": table.get(raw_label, {}),
            "language": language,
        }
    if table and set(table) != {n.taxon.label for n in tree.leaf_node_iter()}:
        raise ValueError("Las puntas y la tabla de metadata no coinciden")
    return {
        "entry": entry,
        "nodes": nodes,
        "root_age_years": crown * 1000,
        "tip_depth_spread": max(tip_depths) - min(tip_depths),
        "metadata": metadata,
    }


def load_tree(entry):
    """Carga solo el árbol solicitado y, si existe, su tabla temporal completa."""
    if entry["kind"] == "mcc":
        text, translations = mcc_source()
        return serialize_tree(parse_tree(text), entry, translations=translations)
    table = {}
    if "metadata" in entry:
        rows = list(csv.DictReader(io.StringIO(read_member(entry["metadata"]))))
        table = {row["tips"]: row for row in rows}
        if len(table) != len(rows):
            raise ValueError("Identificadores duplicados en metadata")
        for row in rows:
            if (
                int(float(row["tree"])) != entry["sample"]
                or abs(abs(float(row["Time2"])) - entry["years"]) > 0.01
            ):
                raise ValueError("La tabla no corresponde a muestra/tiempo")
            if (
                abs((float(row["Crown"]) - float(row["Time"])) * 1000 - entry["years"])
                > 0.1
            ):
                raise ValueError("Unidades temporales inesperadas")
    return serialize_tree(parse_tree(read_member(entry["tree"])), entry, table)

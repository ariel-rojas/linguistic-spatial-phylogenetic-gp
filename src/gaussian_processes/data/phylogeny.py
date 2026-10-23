"""Árbol MCC fijado por hash y covarianza de tiempos compartidos en milenios.

Se conserva la raíz original y cada longitud publicada. El archivo contiene dos
copias idénticas en topología/ramas: STATE_1 anotada y STATE_0 sin anotaciones.
El adaptador de este NEXUS resuelve Translate explícitamente y delega la lectura
Newick a Bio.Phylo; no interpreta soporte ni coordenadas como longitudes.
"""

import gzip
import hashlib
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from Bio import Phylo

from ..configuration import ROOT
from ..models.phylogenetic_covariance import PhylogeneticCovariance

TREE_NAME = "global-language-summary-tree-relabelled_cartoonised.tree.gz"
TREE_URL = (
    "https://github.com/rbouckaert/global-language-tree-pipeline/releases/"
    "download/v2.0.0/" + TREE_NAME
)
TREE_SHA256 = "35578af18d987e58172c88793b33e8ba60626404bdad246610cd94e73146cb46"
MANIFEST = ROOT / "datos/phylogeny/manifest.json"


def file_hash(path):
    """Huella del archivo completo para detectar desalineación o cambios."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tree(path):
    """Lee STATE_1 de la fuente fijada; comprueba su duplicado sin anotaciones.

    Bio.Phylo/NexusIO no resuelve Translate en este archivo tal como se publica.
    Extraemos esa tabla y el Newick sin comentarios; Bio.Phylo lee las ramas.
    El original permanece intacto. Este adaptador es específico de esta release.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        text = handle.read()
    clean = re.sub(r"\[[^\[\]]*\]", "", text)
    translated = re.search(r"(?is)\btranslate\s+(.*?);", clean)
    if translated is None:
        raise ValueError("Falta la tabla Translate del árbol")
    labels = dict(re.findall(r"(\d+)\s+([^,\s]+)", translated.group(1)))
    trees = dict(re.findall(r"(?i)\btree\s+(\w+)\s*=\s*([^;]+);", clean))
    if set(trees) != {"STATE_1", "STATE_0"} or trees["STATE_1"] != trees["STATE_0"]:
        raise ValueError("El archivo no coincide con las dos copias MCC esperadas")
    tree = Phylo.read(io.StringIO(trees["STATE_1"] + ";"), "newick")
    tree.name, tree.rooted = "STATE_1", True  # Raíz original; nunca se re-enraíza.
    for tip in tree.get_terminals():
        label = labels.get(tip.name, "")
        if not re.match(r"^[a-z]{4}[0-9]{4}_", label):
            raise ValueError(f"Hoja sin glottocode: {tip.name}")
        tip.name = label[:8]
    return tree


def build_shared_time_covariance(tree, names):
    """Suma cada rama al bloque de sus descendientes, preservando la raíz.

    names fija el orden de filas/columnas. No normaliza ni invierte la matriz.
    La raíz tiene estado fijo y rama entrante cero; las ramas terminales aportan
    varianza individual. Ramas no negativas garantizan semidefinición positiva.
    """
    tips = tree.get_terminals()
    available = [tip.name for tip in tips]
    if len(set(available)) != len(available) or len(set(names)) != len(names):
        raise ValueError("Identificadores de hojas repetidos")
    if not set(names) <= set(available) or not names:
        raise ValueError("La matriz necesita hojas conocidas y únicas")
    if tree.root.branch_length not in (None, 0):
        raise ValueError("La raíz original debe tener rama entrante cero")
    indices = {name: i for i, name in enumerate(names)}
    covariance = np.zeros((len(names), len(names)))
    descendants = {}
    for node in tree.find_clades(order="postorder"):
        if node is tree.root:
            continue
        length = node.branch_length
        if length is None or not np.isfinite(length) or length < 0:
            raise ValueError("Rama ausente, negativa o no finita")
        if node.is_terminal():
            selected = [indices[node.name]] if node.name in indices else []
        else:
            selected = [i for child in node.clades for i in descendants.pop(child)]
        descendants[node] = selected
        covariance[np.ix_(selected, selected)] += length
    depths = tree.depths()
    by_name = {tip.name: depths[tip] for tip in tips}
    if not np.allclose(covariance.diagonal(), [by_name[name] for name in names]):
        raise ValueError("La diagonal no coincide con las profundidades originales")
    return covariance


def prepare_phylogeny(df):
    """Descarga/audita el MCC y enlaza WALS por glottocode exacto, sin usar 83A."""
    raw = ROOT / "datos/originales/phylogeny" / TREE_NAME
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        with urlopen(TREE_URL, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != TREE_SHA256:
            raise ValueError("Hash inesperado al descargar el árbol")
        raw.write_bytes(content)
    if file_hash(raw) != TREE_SHA256:
        raise ValueError("El árbol local no coincide con la release fijada")
    tree = read_tree(raw)
    tips = tree.get_terminals()
    if len(tips) != 6636:
        raise ValueError("Se esperaban 6636 hojas")
    names = sorted(set(df.Glottocode) & {tip.name for tip in tips})
    covariance = build_shared_time_covariance(tree, names)
    # Una comprobación numérica complementaria, sin reparar ni alterar C.
    minimum = float(np.linalg.eigvalsh(covariance)[0])
    if minimum < -1e-8 * covariance.diagonal().max():
        raise ValueError("Covarianza no semidefinida positiva")
    folder = MANIFEST.parent
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "covariance.npy", covariance)
    pd.DataFrame(
        {"tree_tip_index": range(len(names)), "tree_glottocode": names}
    ).to_csv(folder / "tips.csv", index=False)
    manifest = {
        "release": "v2.0.0",
        "url": TREE_URL,
        "tree_name": "STATE_1",
        "duplicate_tree": "STATE_0 has identical topology and branch lengths",
        "prepared_at": datetime.now(UTC).isoformat(),
        "units": "thousand_years",
        "covariance": "brownian_shared_time_no_normalization",
        "root_state": "fixed",
        "n_tree_tips": len(tips),
        "n_matched_tips": len(names),
        "root_to_tip_min": float(covariance.diagonal().min()),
        "root_to_tip_max": float(covariance.diagonal().max()),
        "minimum_eigenvalue": minimum,
        "files": {
            str(path.relative_to(ROOT)).replace(chr(92), "/"): file_hash(path)
            for path in [raw, folder / "covariance.npy", folder / "tips.csv"]
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    mapping = {name: i for i, name in enumerate(names)}
    df["tree_tip_index"] = df.Glottocode.map(mapping).fillna(-1).astype(int)
    df["tree_glottocode"] = df.Glottocode.where(df.tree_tip_index.ge(0), "")
    return df


def align_tip_families(tips, data):
    """Una familia por hoja: une alias, rechaza conflictos y aísla desconocidas.

    Usa exclusivamente metadatos de identidad, nunca etiquetas OV/VO. Una hoja
    con una asignación conocida y alias vacíos conserva la asignación conocida.
    Una lengua aislada o sin familia no se agrupa con otras desconocidas.
    """
    if data is None or "familia_id" not in data:
        raise ValueError("El bloqueo familiar requiere la tabla de lenguas completa")
    assignments = data[["Glottocode", "familia_id"]].fillna("").copy()
    assignments["familia_id"] = assignments.familia_id.astype(str).str.strip()
    known = assignments.loc[
        assignments.familia_id.ne("")
        & assignments.Glottocode.isin(tips.tree_glottocode)
    ]
    grouped = known.groupby("Glottocode").familia_id.agg(lambda values: sorted(set(values)))
    conflicts = grouped[grouped.map(len).gt(1)]
    if len(conflicts):
        raise ValueError(f"Familias contradictorias por hoja: {conflicts.index.tolist()}")
    mapping = grouped.map(lambda values: values[0]).to_dict()
    labels = [
        "family:" + mapping[tip] if tip in mapping else "tip:" + tip
        for tip in tips.tree_glottocode
    ]
    return labels


def load_phylogeny(config, data=None):
    """Valida artefactos y correspondencia con la tabla antes de ajustar un GP."""
    path = ROOT / config["phylogeny_manifest"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        if file_hash(ROOT / filename) != expected:
            raise ValueError(
                f"Cambió el artefacto filogenético: {filename}; ejecuta prepare"
            )
    covariance = np.load(path.parent / "covariance.npy", allow_pickle=False)
    tips = pd.read_csv(path.parent / "tips.csv", keep_default_na=False)
    if covariance.shape != (len(tips), len(tips)) or not np.isfinite(covariance).all():
        raise ValueError("Matriz filogenética inválida")
    if not np.array_equal(tips.tree_tip_index, np.arange(len(tips))):
        raise ValueError("Orden de hojas inválido")
    if data is not None:
        mapping = dict(zip(tips.tree_glottocode, tips.tree_tip_index))
        expected = data.Glottocode.map(mapping).fillna(-1).astype(int)
        if not np.array_equal(expected, data.tree_tip_index):
            raise ValueError("Tabla de lenguas y matriz desalineadas; ejecuta prepare")
    policy = PhylogeneticCovariance(
        config.get("phylogenetic_structure", "global"),
        config.get("phylogenetic_normalize", False),
    )
    families = align_tip_families(tips, data) if policy.structure == "within_family" else None
    covariance = policy.transform(covariance, families)
    # El manifiesto original describe C cruda. Este apartado registra la matriz
    # efectivamente usada, incluida la política de familias desconocidas.
    manifest["model_covariance"] = {
        "structure": policy.structure,
        "normalized": policy.normalize,
        "units": "correlation" if policy.normalize else "thousand_years",
        "sha256": hashlib.sha256(covariance.astype("<f8").tobytes()).hexdigest(),
        "unknown_family_policy": "singleton_by_tip" if families is not None else None,
        "unknown_family_tips": sum(label.startswith("tip:") for label in families) if families is not None else 0,
        "family_assignment_sha256": hashlib.sha256(json.dumps(families).encode()).hexdigest() if families is not None else None,
    }
    return covariance, manifest

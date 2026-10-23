"""Prepara WALS, audita identidades con low y enlaza la filogenia MCC."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

import low
import pandas as pd

from ..configuration import ROOT
from .phylogeny import prepare_phylogeny

DATA = ROOT / "datos"
REV = "81298aa26d01fdc11b769bcc7fa87ad9e43d7ccd"


def download_sources():
    """Reutiliza la descarga local y registra revisión WALS, versión low y hashes."""
    raw = DATA / "originales"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {
        "wals_revision": REV,
        "low_version": version("languages-of-the-world"),
        "archivos": {},
    }
    for name in [
        "languages.csv",
        "values.csv",
        "codes.csv",
        "StructureDataset-metadata.json",
    ]:
        url = f"https://raw.githubusercontent.com/cldf-datasets/wals/{REV}/cldf/{name}"
        path = raw / name
        if not path.exists():
            with urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        manifest["archivos"][name] = {
            "url": url,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    low_path = Path(low.__file__).parent / "data" / "low_db.json"
    manifest["low_db_sha256"] = hashlib.sha256(low_path.read_bytes()).hexdigest()
    (DATA / "fuentes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return raw


def read_wals(raw):
    """Une lenguas con 83A; preserva sin dominante como categoría observada."""
    languages = pd.read_csv(raw / "languages.csv", keep_default_na=False)
    values = pd.read_csv(raw / "values.csv", keep_default_na=False)
    codes = pd.read_csv(raw / "codes.csv", keep_default_na=False)
    target = values.loc[
        values.Parameter_ID == "83A", ["Language_ID", "Code_ID", "Source"]
    ]
    assert target.Language_ID.is_unique, (
        "Revisar múltiples observaciones de 83A por lengua"
    )
    print(codes.loc[codes.Parameter_ID == "83A"].to_string(index=False))
    # Los registros de géneros/familias del CLDF no son lenguas.
    languages = languages.loc[
        ~languages.ID.str.startswith(("genus-", "family-"))
    ].copy()
    df = languages.merge(
        target, left_on="ID", right_on="Language_ID", how="left", validate="one_to_one"
    )
    df["observado"] = df.Code_ID.map(
        {"83A-1": "OV", "83A-2": "VO", "83A-3": "Sin dominante"}
    ).fillna("Faltante")
    assert set(target.Code_ID) <= {"83A-1", "83A-2", "83A-3"}
    return df


def link_genealogy(df):
    """Enlaza por glottocode único o ISO individual cuando WALS no tiene glottocode.

    No resuelve nombres ni inventa ancestros. Los conflictos quedan registrados.
    """
    db = low.LanguagesOfTheWorld()
    by_gc = {}
    for lang in db.languages:
        if lang.glottocode:
            by_gc.setdefault(lang.glottocode, []).append(lang)
    links = []
    for row in df.itertuples():
        candidates = by_gc.get(row.Glottocode, [])
        lang = candidates[0] if len(candidates) == 1 else None
        match = "glottocode" if lang else "sin_enlace"
        iso_lang = db.languages.get(row.ISO639P3code) if row.ISO639P3code else None
        # No sustituimos una variedad con glottocode distinto por su macrolengua.
        if not row.Glottocode and iso_lang and iso_lang.scope == "I":
            lang, match = iso_lang, "iso_sin_glottocode_wals"
        nodes = []
        node = lang.family if lang else None
        seen = set()
        while node:
            assert node.glottocode not in seen, "Ciclo en la clasificación"
            seen.add(node.glottocode)
            nodes.append(node)
            node = node.parent
        links.append(
            {
                "enlace": match,
                "low_iso": lang.part3 if lang else "",
                "familia_low": nodes[-1].label if nodes else "",
                "familia_id": nodes[-1].glottocode if nodes else "",
                "conflicto_iso": bool(
                    lang and iso_lang and lang.part3 != iso_lang.part3
                ),
            }
        )
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(links)], axis=1)
    return df, str(db)


def validate_metadata(df):
    """Marca coordenadas válidas sin eliminar filas de la auditoría."""
    for col in ["Latitude", "Longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["coordenadas_validas"] = df.Latitude.between(-90, 90) & df.Longitude.between(
        -180, 180
    )
    return df


def group_identifiers(df):
    """Une transitivamente filas que comparten ISO o glottocode para evitar filtraciones."""
    # Agrupación conservadora: toda cadena de identificadores compartidos va junta.
    parent = {i: i for i in df.index}

    def find(i):
        """Encuentra el representante y comprime la cadena de identidades unidas."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    identifiers = {}
    for i, row in df.iterrows():
        for col in ["Glottocode", "ISO639P3code", "low_iso"]:
            token = row[col]
            key = ("iso" if col != "Glottocode" else "gc", token)
            if token:
                if key in identifiers:
                    parent[find(i)] = find(identifiers[key])
                identifiers[key] = i
    df["unidad"] = [df.loc[find(i), "ID"] for i in df.index]
    return df


def prepare_data():
    """Ejecuta la preparación y guarda tabla auditable y resumen de cobertura."""
    raw = download_sources()
    df = read_wals(raw)
    df, low_summary = link_genealogy(df)
    df = validate_metadata(df)
    df = prepare_phylogeny(df)
    df = group_identifiers(df)
    df["elegible_modelo"] = (
        df.coordenadas_validas & df.tree_tip_index.ge(0) & ~df.conflicto_iso
    )
    df["motivo_exclusion"] = [
        "; ".join(
            reason
            for flag, reason in [
                (not row.coordenadas_validas, "sin coordenadas"),
                (row.tree_tip_index < 0, "sin hoja en el árbol MCC"),
                (row.conflicto_iso, "conflicto de identificadores"),
            ]
            if flag
        )
        for row in df.itertuples()
    ]
    df.to_csv(DATA / "lenguas.csv", index=False)
    audit = {
        "low": low_summary,
        "lenguas_wals": len(df),
        "observado": df.observado.value_counts().to_dict(),
        "enlaces": df.enlace.value_counts().to_dict(),
        "sin_coordenadas": int((~df.coordenadas_validas).sum()),
        "conflictos_iso": int(df.conflicto_iso.sum()),
        "elegibles_modelo": int(df.elegible_modelo.sum()),
        "con_hoja_arbol": int(df.tree_tip_index.ge(0).sum()),
        "elegibles_por_clase": df.loc[df.elegible_modelo, "observado"]
        .value_counts()
        .to_dict(),
    }
    (DATA / "auditoria.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))

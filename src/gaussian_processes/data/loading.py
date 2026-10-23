"""Lectura común de los datos auditados; conserva su esquema original en español."""

import pandas as pd

from ..configuration import ROOT


def load_data():
    """Lee la tabla WALS + low, sin modificar observaciones ni motivos de exclusión."""
    df = pd.read_csv(ROOT / "datos/lenguas.csv", keep_default_na=False)
    if "tree_tip_index" not in df:
        raise ValueError(
            "Datos anteriores al árbol MCC: ejecuta gaussian-processes prepare"
        )
    for column in ["Latitude", "Longitude"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

"""Coordenadas espaciales e identificadores estables de hojas del árbol."""

import numpy as np


def build_features(df):
    """Devuelve xyz + índice global de hoja, sin usar etiquetas OV/VO."""
    if (df.tree_tip_index < 0).any():
        raise ValueError("Una lengua sin hoja no puede entrar al modelo combinado")
    return np.column_stack(
        [coordinates_xyz(df.Latitude, df.Longitude), df.tree_tip_index]
    )


def coordinates_xyz(latitude, longitude):
    """Puntos en una esfera de radio 6371 km; evita el corte del antimeridiano."""
    lat, lon = np.radians(latitude), np.radians(longitude)
    return 6371 * np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )

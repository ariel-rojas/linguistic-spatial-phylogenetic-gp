"""Particiones por identidad: ninguna identidad cruza una separación."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ..configuration import CLASSES


def validate_split(df, train, test):
    """Exige conjuntos no vacíos y grupos disjuntos."""
    if not len(train) or not len(test):
        raise ValueError("La partición necesita entrenamiento y prueba no vacíos")
    if set(df.iloc[train].unidad) & set(df.iloc[test].unidad):
        raise ValueError("Identificadores compartidos cruzan la partición")


def split_data(df, config, split_type="random"):
    """Reserva grupos de forma estratificada o una familia completa; devuelve posiciones.

    El split aleatorio aproxima proporciones de clases, sin bloquear familias/regiones.
    test_fraction debe ser el inverso de un entero: 0.2 reserva una de 5 particiones.
    Una unidad reúne transitivamente registros con identificadores compartidos.
    """
    if split_type == "random":
        fold_count = round(1 / config["test_fraction"])
        if fold_count < 2 or not np.isclose(1 / fold_count, config["test_fraction"]):
            raise ValueError("test_fraction debe ser 1/k (por ejemplo 0.2, 0.25 o 0.1)")
        splitter = StratifiedGroupKFold(
            n_splits=fold_count, shuffle=True, random_state=config["seed"]
        )
        train, test = next(splitter.split(df, df.observado, groups=df.unidad))
        if set(df.iloc[train].observado) != set(CLASSES) or set(
            df.iloc[test].observado
        ) != set(CLASSES):
            raise ValueError(
                "La partición externa debe contener las tres clases en ambos conjuntos"
            )
    elif split_type == "family":
        selected = df.familia_id.eq(config["test_family"]).to_numpy()
        train, test = (
            np.flatnonzero(~selected & df.familia_id.ne("").to_numpy()),
            np.flatnonzero(selected),
        )
    else:
        raise ValueError(f"Partición desconocida: {split_type}")
    validate_split(df, train, test)
    return train, test


def inner_splits(training_rows, config):
    """CV solo dentro del entrenamiento externo, estratificada y agrupada.

    Se exige que cada entrenamiento y validación contenga las tres clases.
    """
    splitter = StratifiedGroupKFold(
        n_splits=config["cv_folds"], shuffle=True, random_state=config["seed"]
    )
    folds = list(
        splitter.split(training_rows, training_rows.observado, training_rows.unidad)
    )
    for train, validation in folds:
        validate_split(training_rows, train, validation)
        for indices in (train, validation):
            if set(training_rows.iloc[indices].observado) != set(CLASSES):
                raise ValueError(
                    "Cada partición interna necesita tres clases; reduce cv_folds"
                )
    return folds

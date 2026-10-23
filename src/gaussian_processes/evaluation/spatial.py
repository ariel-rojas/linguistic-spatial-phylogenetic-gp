"""Validación geográfica con bloques y separación mínima, sin selección global.

El cubrimiento xyz evita una discontinuidad especial en el antimeridiano. Los
bloques son intersecciones esfera/cubo, no regiones de igual área ni países.
El buffer usa distancia superficial y excluye identidades completas de train.
"""

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ..configuration import CLASSES, PROBABILITY_COLUMNS, ROOT, load_configuration
from ..data.loading import load_data
from ..data.phylogeny import load_phylogeny
from ..models.features import build_features
from ..models.gp import build_model
from ..models.spherical import EARTH_RADIUS_KM, great_circle_distance
from .experiment import prediction_rows, write_json
from .metrics import classification_metrics
from .splits import validate_split


def spatial_groups(data, X, block_km):
    """Une celdas que comparten identidad: ninguna unidad puede cruzar folds."""
    if not np.isfinite(block_km) or block_km <= 0:
        raise ValueError("block_km debe ser positivo y finito")
    cells = np.floor((X[:, :3] + EARTH_RADIUS_KM) / block_km).astype(int)
    keys = [tuple(row) for row in cells]
    parents = {key: key for key in keys}

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    first = {}
    for unit, cell in zip(data.unidad, keys):
        if unit in first:
            parents[root(cell)] = root(first[unit])
        else:
            first[unit] = cell
    return np.array(["_".join(map(str, root(key))) for key in keys])


def spatial_splits(data, X, folds=3, block_km=2000, buffer_km=250):
    """Devuelve train, test y purgados por fold; cada observación es test una vez.

    El buffer asegura que ningún punto de train quede a menos de buffer_km de
    un punto test. La estratificación de clases no se impone a la geografía.
    """
    if not np.isfinite(buffer_km) or buffer_km < 0:
        raise ValueError("buffer_km debe ser finito y no negativo")
    groups = spatial_groups(data, X, block_km)
    if len(set(groups)) < folds:
        raise ValueError("Pocos bloques: reduce block_km o cv_folds")
    result = []
    for train, test in GroupKFold(n_splits=folds).split(X, groups=groups):
        distances = great_circle_distance(X[train, :3], X[test, :3])
        near_units = set(data.iloc[train[np.min(distances, axis=1) < buffer_km]].unidad)
        purge_mask = data.iloc[train].unidad.isin(near_units).to_numpy()
        purged, train = train[purge_mask], train[~purge_mask]
        validate_split(data, train, test)
        if set(data.iloc[train].observado) != set(CLASSES):
            raise ValueError(
                "Train espacial sin las tres clases; revisa bloques y buffer"
            )
        result.append((train, test, purged))
    return groups, result


def evaluate_spatial(filename, block_km=2000, buffer_km=250):
    """Evalúa valores principales del JSON; nunca reutiliza el ganador aleatorio."""
    config, folder = load_configuration(filename)
    out = folder / "spatial"
    out.mkdir(parents=True, exist_ok=True)
    data = load_data()
    covariance, phylogeny = load_phylogeny(config, data)
    data = data.loc[data.elegible_modelo & data.observado.isin(CLASSES)].reset_index(
        drop=True
    )
    X = build_features(data)
    groups, folds = spatial_splits(data, X, config["cv_folds"], block_km, buffer_km)
    predictions, metrics = [], []
    for number, (train, test, purged) in enumerate(folds, start=1):
        print(
            f"Bloque/fold {number}/{len(folds)}: {len(train)} train, {len(test)} test, {len(purged)} excluidos por buffer",
            flush=True,
        )
        model = build_model(config, covariance).fit(
            X[train], data.iloc[train].observado
        )
        rows = prediction_rows(data.iloc[test], model, X[test])
        rows["spatial_fold"], rows["spatial_block"] = number, groups[test]
        predictions.append(rows)
        metrics.append(
            {
                "fold": number,
                "n_train": len(train),
                "purged_ids": data.iloc[purged].ID.tolist(),
                **classification_metrics(rows.observado, rows[PROBABILITY_COLUMNS]),
            }
        )
    predictions = pd.concat(predictions, ignore_index=True)
    assert predictions.ID.is_unique and set(predictions.ID) == set(data.ID)
    predictions.to_csv(out / "predictions.csv", index=False)
    write_json(
        out / "run.json",
        {
            "configuration": config,
            "selection": "fixed_parameters_no_search",
            "block_km": block_km,
            "buffer_km": buffer_km,
            "block_definition": "Cubos xyz unidos por identidad; escala aproximada, no área uniforme",
            "folds": metrics,
            "pooled_by_language": classification_metrics(
                predictions.observado, predictions[PROBABILITY_COLUMNS]
            ),
            "phylogeny": phylogeny,
            "data_sha256": hashlib.sha256(
                (ROOT / "datos/lenguas.csv").read_bytes()
            ).hexdigest(),
        },
    )
    print(f"Resultados espaciales: {out}", flush=True)

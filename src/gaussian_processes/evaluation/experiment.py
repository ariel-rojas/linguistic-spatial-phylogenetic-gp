"""Evaluación independiente e imputación posterior con todas las etiquetas conocidas."""

import hashlib
import json
from importlib.metadata import version

import numpy as np
import pandas as pd

from ..configuration import CLASSES, PROBABILITY_COLUMNS, ROOT, load_configuration
from ..data.loading import load_data
from ..data.phylogeny import load_phylogeny
from ..models.features import build_features
from ..models.gp import build_model, predict
from .metrics import classification_metrics
from .splits import split_data


def write_json(path, content):
    """Guarda JSON legible, sin NaN no estándar."""
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def prediction_rows(df, model, X):
    """Añade las tres probabilidades y la etiqueta predicha a una copia de la tabla."""
    probabilities, labels = predict(model, X)
    rows = df.copy()
    rows[PROBABILITY_COLUMNS] = probabilities
    rows["predicho"] = labels
    return rows


def evaluate(df, X, config, covariance, split_type="random", model=None):
    """Evalúa solo test; acepta el estimador ya reajustado por GridSearchCV.

    model, si se proporciona, debe haberse entrenado exclusivamente con el train
    de esta misma partición. El flujo de búsqueda garantiza esa condición.
    """
    if split_type == "family":
        keep = df.familia_id.ne("").to_numpy()
        df, X = df.loc[keep].copy(), X[keep]
    train, test = split_data(df, config, split_type)
    if set(df.iloc[train].observado) != set(CLASSES):
        raise ValueError("Entrenamiento necesita OV, VO y Sin dominante")
    if model is None:
        model = build_model(config, covariance).fit(X[train], df.iloc[train].observado)
    tested = prediction_rows(df.iloc[test], model, X[test])
    metrics = classification_metrics(tested.observado, tested[PROBABILITY_COLUMNS])
    metrics.update(
        {
            "split": split_type,
            "n_train": len(train),
            "test_family": config["test_family"] if split_type == "family" else None,
        }
    )
    rows = df.copy()
    rows[PROBABILITY_COLUMNS] = np.nan
    rows["predicho"], rows["estado"] = "", "Entrenamiento"
    rows.loc[tested.index, PROBABILITY_COLUMNS + ["predicho"]] = tested[
        PROBABILITY_COLUMNS + ["predicho"]
    ]
    rows.loc[tested.index, "estado"] = np.where(
        tested.observado == tested.predicho, "Correcto", "Incorrecto"
    )
    return rows, metrics


def impute(df, X, config, covariance):
    """Reajusta con todas las clases observadas; las imputaciones no son validación."""
    known = df.observado.isin(CLASSES).to_numpy()
    missing = df.observado.eq("Faltante").to_numpy()
    model = build_model(config, covariance).fit(X[known], df.loc[known, "observado"])
    rows = prediction_rows(df.loc[missing], model, X[missing])
    rows["estado"] = "Imputado sin verificar"
    return rows


def save_run(
    out,
    config,
    selected,
    evaluation,
    metrics,
    eligible,
    X,
    covariance,
    phylogeny,
    search_summary=None,
):
    """Guarda evaluación antes del ajuste final, e identifica datos y versiones."""
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "run.json",
        {
            "configuration": config,
            "selected_parameters": selected,
            "metrics": metrics,
            "search": search_summary,
            "data_sha256": hashlib.sha256(
                (ROOT / "datos/lenguas.csv").read_bytes()
            ).hexdigest(),
            "versions": {
                name: version(name)
                for name in [
                    "languages-of-the-world",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "scipy",
                    "biopython",
                ]
            },
            "classes": CLASSES,
            "phylogeny": phylogeny,
            "selection": "inner_grouped_cv"
            if out.name != "fixed"
            else "fixed_parameters",
        },
    )
    print(
        "Evaluación guardada. Ajustando con todos los datos para imputar...", flush=True
    )
    imputed = impute(eligible, X, selected, covariance)
    data = load_data()
    # Una fila por registro WALS. El rol distingue evaluación, ajuste e imputación.
    excluded = data.loc[~data.ID.isin(pd.concat([evaluation.ID, imputed.ID]))].copy()
    excluded["estado"] = np.where(
        excluded.observado.eq("Faltante"), "Faltante sin imputar", "Observado excluido"
    )
    rows = pd.concat([evaluation, imputed, excluded], ignore_index=True)
    rows["role"] = rows.estado.map(
        {
            "Entrenamiento": "train",
            "Correcto": "test",
            "Incorrecto": "test",
            "Imputado sin verificar": "imputed",
            "Faltante sin imputar": "excluded",
            "Observado excluido": "excluded",
        }
    )
    assert rows.ID.is_unique and set(rows.ID) == set(data.ID)
    rows.to_csv(out / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"Resultados: {out}", flush=True)


def train(filename):
    """Ejecuta una referencia de parámetros fijos en una carpeta independiente."""
    config, folder = load_configuration(filename)
    out = folder / "fixed"
    data = load_data()
    covariance, phylogeny = load_phylogeny(config, data)
    eligible = data.loc[data.elegible_modelo].copy().reset_index(drop=True)
    X = build_features(eligible)
    known = eligible.observado.isin(CLASSES).to_numpy()
    rows, metrics = evaluate(eligible.loc[known], X[known], config, covariance)
    save_run(out, config, config, rows, metrics, eligible, X, covariance, phylogeny)
    if config["test_family"] is not None:
        rows, metrics = evaluate(
            eligible.loc[known], X[known], config, covariance, "family"
        )
        rows.loc[rows.estado.ne("Entrenamiento")].to_csv(
            out / "family_evaluation.csv", index=False
        )
        summary = json.loads((out / "run.json").read_text(encoding="utf-8"))
        summary["held_out_family_metrics"] = metrics
        write_json(out / "run.json", summary)

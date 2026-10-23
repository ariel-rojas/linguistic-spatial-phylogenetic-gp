"""Distribución al reservar cada familia; parámetros fijos, sin selección global."""

import hashlib

import pandas as pd

from ..configuration import CLASSES, PROBABILITY_COLUMNS, ROOT, load_configuration
from ..data.loading import load_data
from ..data.phylogeny import load_phylogeny
from ..models.features import build_features
from ..visualization.distribution import plot_distribution
from .experiment import evaluate, write_json
from .metrics import classification_metrics


def summarize(metrics, predictions):
    """Separa promedios por familia de métricas agrupadas por lengua.

    Las familias pequeñas tienen el mismo peso en la primera media. Los percentiles
    describen heterogeneidad, no son intervalos de confianza. El recall balanceado
    de cada familia solo cubre sus clases presentes; no equivale a cubrir tres clases.
    """
    columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "brier_multiclass",
    ]
    distribution = metrics[columns].describe(percentiles=[0.25, 0.5, 0.75])
    summary = {
        "n_families": len(metrics),
        "n_languages": len(predictions),
        "mean_by_family": metrics[columns].mean().to_dict(),
        "pooled_by_language": classification_metrics(
            predictions.observado, predictions[PROBABILITY_COLUMNS]
        ),
    }
    return summary, distribution


def evaluate_all_families(filename):
    """Reajusta un GP por familia con los valores principales del JSON.

    No reutiliza los parámetros seleccionados con el split aleatorio: esa selección
    habría visto etiquetas de algunas familias que aquí deben quedar reservadas.
    Una selección por familia requeriría CV anidada dentro de cada entrenamiento.
    """
    config, folder = load_configuration(filename)
    out = folder / "families"
    out.mkdir(parents=True, exist_ok=True)
    data = load_data()
    covariance, phylogeny = load_phylogeny(config, data)
    unknown = (
        data.elegible_modelo & data.observado.isin(CLASSES) & data.familia_id.eq("")
    )
    df = (
        data.loc[
            data.elegible_modelo & data.observado.isin(CLASSES) & data.familia_id.ne("")
        ]
        .copy()
        .reset_index(drop=True)
    )
    X = build_features(df)
    families = sorted(df.familia_id.unique())
    if len(families) < 2:
        raise ValueError("Se necesitan al menos dos familias")
    fold_metrics, predictions, per_class = [], [], []
    for number, family in enumerate(families, start=1):
        print(f"Familia {number}/{len(families)}: {family}", flush=True)
        rows, metrics = evaluate(
            df, X, dict(config, test_family=family), covariance, "family"
        )
        tested = rows.loc[rows.estado.ne("Entrenamiento")].copy()
        tested["held_out_family"] = family
        per_class.extend(
            {"test_family": family, "class": label, **values}
            for label, values in metrics["per_class"].items()
        )
        fold_metrics.append(
            {
                key: value
                for key, value in metrics.items()
                if key not in ["classes", "per_class", "confusion_matrix"]
            }
        )
        predictions.append(tested)
    metrics = pd.DataFrame(fold_metrics)
    predictions = pd.concat(predictions, ignore_index=True)
    assert predictions.ID.is_unique and set(predictions.ID) == set(df.ID)
    summary, distribution = summarize(metrics, predictions)
    metrics.to_csv(out / "metrics_by_family.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    write_json(
        out / "run.json",
        {
            **summary,
            "distribution": distribution.to_dict(),
            "per_family_class": per_class,
            "excluded_unknown_family_ids": data.loc[unknown, "ID"].tolist(),
            "configuration": config,
            "phylogeny": phylogeny,
            "n_excluded_unknown_family": int(unknown.sum()),
            "selection": "fixed_parameters_no_search",
            "data_sha256": hashlib.sha256(
                (ROOT / "datos/lenguas.csv").read_bytes()
            ).hexdigest(),
        },
    )
    plot_distribution(metrics, out)
    print(f"Resultados: {out}", flush=True)

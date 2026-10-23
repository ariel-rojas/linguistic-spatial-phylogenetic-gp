"""Dos ajustes independientes: evaluar sobre train y después imputar con todos."""

import hashlib
import json
from importlib.metadata import version

import numpy as np

from ..configuration import CLASSES, PROBABILITY_COLUMNS, ROOT, load_configuration
from ..data.loading import load_data
from ..data.phylogeny import load_phylogeny
from ..evaluation.experiment import write_json
from ..evaluation.metrics import classification_metrics
from ..evaluation.splits import split_data
from .decomposition import predictive_components, total_variation
from .design import phylogenetic_design, spherical_basis
from .inference import (
    flatten_samples,
    prior_predictive,
    probability_draws,
    sample_posterior,
)


def load_sampling_settings(filename):
    """Valida controles de muestreo y priors; no se usa la grilla de sklearn."""
    settings = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    expected = {
        "chains",
        "warmup",
        "draws",
        "target_accept",
        "max_tree_depth",
        "amplitude_scale",
        "intercept_scale",
        "smoothing_median",
        "smoothing_log_sd",
        "prior_draws",
        "predictive_draws",
        "seed",
    }
    if set(settings) != expected:
        raise ValueError(f"Claves de muestreo requeridas: {sorted(expected)}")
    for key, value in settings.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} debe ser positivo y finito")
    for key in [
        "chains",
        "warmup",
        "draws",
        "max_tree_depth",
        "prior_draws",
        "predictive_draws",
        "seed",
    ]:
        if type(settings[key]) is not int:
            raise ValueError(f"{key} debe ser entero")
    if (
        settings["chains"] < 2
        or settings["draws"] < 4
        or not 0 < settings["target_accept"] < 1
    ):
        raise ValueError("Se requieren >=2 cadenas, >=4 draws y 0<target_accept<1")
    return settings


def summarize_predictions(rows, probabilities):
    """Medias e intervalos puntuales de probabilidad, nunca intervalos de etiquetas."""
    result = rows.copy()
    mean = probabilities.mean(axis=0)
    result[PROBABILITY_COLUMNS] = mean
    for index, column in enumerate(PROBABILITY_COLUMNS):
        result[column + "_lower"] = np.quantile(
            probabilities[:, :, index], 0.025, axis=0
        )
        result[column + "_upper"] = np.quantile(
            probabilities[:, :, index], 0.975, axis=0
        )
    result["predicho"] = np.asarray(CLASSES)[mean.argmax(axis=1)]
    entropy = lambda p: -(p * np.log(np.maximum(p, 1e-300))).sum(axis=-1)
    result["predictive_entropy"] = entropy(mean)
    result["epistemic_information"] = np.maximum(
        0, entropy(mean) - entropy(probabilities).mean(axis=0)
    )
    return result


def fit_stage(
    eligible,
    fit_indices,
    target_indices,
    covariance,
    basis,
    degrees,
    settings,
    seed,
    path,
):
    """Ajusta solo fit_indices; target_indices no aporta observaciones al modelo."""
    fit, target = eligible.iloc[fit_indices], eligible.iloc[target_indices]
    factor, mean, residual, unique_tips = phylogenetic_design(
        covariance, fit.tree_tip_index.to_numpy(), target.tree_tip_index.to_numpy()
    )
    labels = np.array([CLASSES.index(label) for label in fit.observado])
    samples, extras, diagnostics = sample_posterior(
        basis[fit_indices], degrees, factor, labels, settings, seed
    )
    selected = flatten_samples(samples, settings["predictive_draws"])
    components = predictive_components(
        selected, basis[target_indices], degrees, mean, residual, seed + 1
    )
    probabilities = components["full"]
    spatial_influence = total_variation(probabilities, components["without_spatial"])
    phylogenetic_influence = total_variation(
        probabilities, components["without_phylogenetic"]
    )
    # Comprobación posterior predictiva en train, nunca usada como métrica test.
    training_p = probability_draws(
        selected, basis[fit_indices], degrees, factor, np.empty((len(fit), 0)), seed + 2
    )
    rng = np.random.default_rng(seed + 3)
    replicated = (
        rng.random(training_p.shape[:2])[..., None] > np.cumsum(training_p, axis=-1)
    ).sum(axis=-1)
    fractions = np.stack([(replicated == c).mean(axis=1) for c in range(3)], axis=-1)
    diagnostics["posterior_predictive_training"] = {
        "observed_class_fractions": [(labels == c).mean().item() for c in range(3)],
        "replicated_fraction_quantiles_025_50_975": np.quantile(
            fractions, [0.025, 0.5, 0.975], axis=0
        ).tolist(),
    }
    imputed_labels = (
        rng.random(probabilities.shape[:2])[..., None]
        > np.cumsum(probabilities, axis=-1)
    ).sum(axis=-1)
    np.savez_compressed(
        path,
        **{"posterior_" + k: v for k, v in samples.items()},
        **{"sampler_" + k: v for k, v in extras.items()},
        probabilities=probabilities.astype(np.float32),
        spatial_influence=spatial_influence.astype(np.float32),
        phylogenetic_influence=phylogenetic_influence.astype(np.float32),
        labels=imputed_labels.astype(np.int8),
        fit_ids=fit.ID.to_numpy(dtype=str),
        target_ids=target.ID.to_numpy(dtype=str),
        unique_training_tips=unique_tips,
        classes=np.asarray(CLASSES),
        settings_json=json.dumps(settings),
        diagnostics_json=json.dumps(diagnostics),
    )
    return summarize_predictions(target, probabilities), diagnostics


def run_bayesian(filename, sampling_filename):
    """Flujo completo con resultados separados; no modifica el análisis sklearn."""
    config, base_folder = load_configuration(filename)
    if config.get("spatial_kernel") != "spherical_spline":
        raise ValueError("El motor bayesiano inicial admite el spline esférico")
    if not config.get("phylogenetic_normalize", False):
        raise ValueError(
            "Los priors de amplitud requieren matriz filogenética normalizada"
        )
    settings = load_sampling_settings(sampling_filename)
    folder = base_folder / "bayesian"
    folder.mkdir(parents=True, exist_ok=True)
    data = load_data()
    covariance, phylogeny = load_phylogeny(config, data)
    eligible = data.loc[data.elegible_modelo].reset_index(drop=True)
    known = np.flatnonzero(eligible.observado.isin(CLASSES))
    missing = np.flatnonzero(eligible.observado.eq("Faltante"))
    train_relative, test_relative = split_data(eligible.iloc[known], config)
    train, test = known[train_relative], known[test_relative]
    basis, degrees = spherical_basis(
        eligible.Latitude, eligible.Longitude, config["max_degree"]
    )
    prior_factor, _, _, _ = phylogenetic_design(
        covariance,
        eligible.iloc[train].tree_tip_index.to_numpy(),
        np.array([], dtype=int),
    )
    summary = {
        "status": "running",
        "configuration": config,
        "sampling": settings,
        "selection": "bayesian_nuts",
        "classes": CLASSES,
        "data_sha256": hashlib.sha256(
            (ROOT / "datos/lenguas.csv").read_bytes()
        ).hexdigest(),
        "phylogeny": phylogeny,
        "versions": {
            name: version(name)
            for name in ["numpyro", "jax", "arviz", "numpy", "scipy"]
        },
        "n_train": len(train),
        "n_test": len(test),
        "n_final_fit": len(known),
        "n_imputed": len(missing),
        "prior_predictive": prior_predictive(
            basis[train], degrees, prior_factor, settings
        ),
        "limitations": [
            "MCC, familias y grado espacial fijos",
            "Test histórico reutilizado: comparación exploratoria",
            "Categórica conjunta distinta de OvR sklearn",
            "Muestreo condicionado a priors; no es prueba de calibración",
        ],
    }
    write_json(folder / "run.json", summary)
    print("Etapa 1/2: evaluar sin observar etiquetas test", flush=True)
    tested, diagnostics = fit_stage(
        eligible,
        train,
        test,
        covariance,
        basis,
        degrees,
        settings,
        settings["seed"],
        folder / "evaluation.npz",
    )
    summary["metrics"] = classification_metrics(
        tested.observado, tested[PROBABILITY_COLUMNS]
    )
    summary["evaluation_diagnostics"] = diagnostics
    write_json(folder / "run.json", summary)
    print(
        json.dumps(
            {"metrics": summary["metrics"], "diagnostics_passed": diagnostics["passed"]}
        ),
        flush=True,
    )
    print(
        "Etapa 2/2: ajuste independiente con todos los observados para imputar",
        flush=True,
    )
    imputed, diagnostics = fit_stage(
        eligible,
        known,
        missing,
        covariance,
        basis,
        degrees,
        settings,
        settings["seed"] + 100,
        folder / "imputation.npz",
    )
    summary["imputation_diagnostics"] = diagnostics
    summary["status"] = (
        "complete"
        if diagnostics["passed"] and summary["evaluation_diagnostics"]["passed"]
        else "diagnostics_warning"
    )
    rows = data.copy().set_index("ID")
    rows["role"] = "excluded"
    rows["estado"] = np.where(
        rows.observado.eq("Faltante"), "Faltante sin imputar", "Observado excluido"
    )
    rows.loc[eligible.iloc[train].ID, ["role", "estado"]] = ["train", "Entrenamiento"]
    for predictions, role in [(tested, "test"), (imputed, "imputed")]:
        predictions = predictions.set_index("ID")
        for column in [
            *PROBABILITY_COLUMNS,
            "predicho",
            "predictive_entropy",
            "epistemic_information",
            *[
                c + suffix
                for c in PROBABILITY_COLUMNS
                for suffix in ["_lower", "_upper"]
            ],
        ]:
            if column not in rows:
                rows[column] = "" if column == "predicho" else np.nan
            rows.loc[predictions.index, column] = predictions[column]
        rows.loc[predictions.index, "role"] = role
        rows.loc[predictions.index, "estado"] = (
            np.where(
                predictions.observado == predictions.predicho, "Correcto", "Incorrecto"
            )
            if role == "test"
            else "Imputado sin verificar"
        )
    rows.reset_index().to_csv(folder / "predictions.csv", index=False)
    write_json(folder / "run.json", summary)
    write_report(folder, summary)
    from ..visualization.map import build_map

    build_map(filename, bayesian=True)
    print(f"Resultado bayesiano: {folder}; estado: {summary['status']}", flush=True)


def write_report(folder, summary):
    """Resumen compacto; siempre distingue ejecución completa y diagnósticos."""
    m = summary["metrics"]
    lines = [
        "# GP bayesiano: resultado",
        "",
        f"Estado: **{summary['status']}**.",
        "",
        f"Test: {m['correct']}/{m['n_test']} aciertos; log-loss {m['log_loss']:.5f}; macro-F1 {m['macro_f1']:.5f}; Brier {m['brier_multiclass']:.5f}.",
        f"Imputaciones sin verificar: {summary['n_imputed']}. No se utilizan para medir aciertos.",
        "",
        "| Ajuste | R-hat máximo | ESS bulk mínimo | ESS tail mínimo | Divergencias | Profundidad máxima alcanzada |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for title, key in [
        ("Train", "evaluation_diagnostics"),
        ("Todos los observados", "imputation_diagnostics"),
    ]:
        d = summary[key]
        lines.append(
            f"| {title} | {d['max_rhat']:.4f} | {d['min_ess_bulk']:.1f} | {d['min_ess_tail']:.1f} | {d['divergences']} | {d['tree_depth_hits']} |"
        )
    lines += [
        "",
        "Si el estado es diagnostics_warning, las cifras son exploratorias y requieren mejorar/revisar el muestreo.",
        "",
        "## Archivos",
        "",
        "- `run.json`: priors, diagnósticos, métricas y trazabilidad.",
        "- `predictions.csv`: roles, probabilidades medias e intervalos creíbles puntuales del 95%.",
        "- `evaluation.npz` y `imputation.npz`: muestras por cadena, probabilidades conjuntas, etiquetas simuladas y sus IDs. Leer con numpy.load(..., allow_pickle=False).",
        "- `map.html`: test, train e imputaciones con intervalos en las fichas.",
        "- `posterior.html`: búsqueda por lengua, histogramas de probabilidades, predictiva categórica y posteriores de parámetros globales.",
        "",
        "El motor es NumPyro/NUTS. No hay grid search ni evidencia de convergencia basada solo en accuracy.",
        "La incertidumbre no incluye árboles alternativos, errores de WALS ni cambios de familia de kernel.",
    ]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
